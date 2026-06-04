"""MM-TSFlib 멀티모달 융합 베이스라인 (원본 메커니즘 충실 재현).

Time-MMD native 베이스라인(MM-TSFlib)의 exp_long_term_forecasting 융합을 그대로 옮김:

    prompt_y = norm(text_mlp(text_emb)) + prior_y          # 텍스트=모양, prior=레벨
    output   = (1 - prompt_weight) * ts_pred + prompt_weight * prompt_y   # 볼록결합

- ts_pred : TS 백본(DLinear/PatchTST) 예측 [B, H, V] (dataset-정규화 공간)
- text_mlp: frozen LLM 임베딩 -> d_llm/8 -> H*V  (원본 MLP [d_llm, d_llm/8, text_emb=pred_len])
- norm()  : pred_len 축 instance-norm (평균0/분산1) — 텍스트는 시퀀스 '모양'만 예측
- prior_y : look-back 윈도우 평균(정규화 공간)을 H로 broadcast — 원본 prior_history_avg 의
            self-contained 대응(과거 평균 레벨 앵커). 텍스트투영이 0이어도 prompt_y≈prior_y.
- 결합    : 볼록결합 (원본 line 461). 단순 additive 가 아니라 ts를 (1-w)로 줄인다.

frozen LLM 임베딩은 data/text_encoder 가 도메인별로 캐시(원본 use_fullmodel=0 의 embedding
layer 풀링과 정렬). trainer 는 is_multimodal 플래그로 forward(x_enc, text_emb) 를 호출한다.
"""
import torch
import torch.nn as nn

from ..config import ModelConfig
from .dlinear import DLinear
from .patchtst import PatchTST
from .itransformer import iTransformer
from .transformer import Transformer
from .autoformer import Autoformer
from .informer import Informer
from .fedformer import FEDformer

_BACKBONES = {
    "DLinear": DLinear, "PatchTST": PatchTST, "iTransformer": iTransformer,
    "Transformer": Transformer, "Autoformer": Autoformer,
    "Informer": Informer, "FEDformer": FEDformer,
}


def _norm_seq(x: torch.Tensor) -> torch.Tensor:
    """MM-TSFlib norm(): 시퀀스(dim=1, pred_len) 축으로 평균0/분산1 정규화."""
    x = x - x.mean(1, keepdim=True).detach()
    x = x / torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    return x


class _MLP(nn.Module):
    """MM-TSFlib MLP: 각 Linear 뒤(마지막 제외) ReLU + Dropout."""

    def __init__(self, sizes, dropout_rate=0.3):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(sizes[i], sizes[i + 1])
                                    for i in range(len(sizes) - 1))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
                x = self.dropout(x)
        return x


class MMFusion(nn.Module):
    is_multimodal = True

    def __init__(self, configs: ModelConfig, backbone: str = "PatchTST"):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}. {list(_BACKBONES)}")
        self.backbone = _BACKBONES[backbone](configs)
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.prompt_weight = configs.prompt_weight
        # 원본 mlp_sizes = [d_llm, d_llm/8, text_emb(=pred_len)]; 다변량은 pred_len*c_out
        hidden = max(configs.d_llm // 8, 1)
        self.text_mlp = _MLP([configs.d_llm, hidden, configs.pred_len * configs.c_out],
                             dropout_rate=0.3)

    def forward(self, x_enc, text_emb):
        ts = self.backbone(x_enc, None, None, None)             # [B, H, V] (정규화 공간)
        B = x_enc.size(0)
        # prior_y: look-back 평균을 H로 broadcast (정규화 공간). 원본 prior_history_avg 대응.
        prior = x_enc.mean(dim=1, keepdim=True).expand(-1, self.pred_len, -1)  # [B, H, V]
        txt = self.text_mlp(text_emb).view(B, self.pred_len, self.c_out)        # [B, H, V]
        prompt_y = _norm_seq(txt) + prior                       # 텍스트=모양 + prior=레벨
        return (1 - self.prompt_weight) * ts + self.prompt_weight * prompt_y
