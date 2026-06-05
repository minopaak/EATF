"""MM-TSFlib 멀티모달 융합 (원본 메커니즘 재현).

  prompt_y = norm(text_mlp(text_emb)) + prior        # 텍스트=시퀀스 모양, prior=레벨
  output   = (1 - prompt_weight) * ts_pred + prompt_weight * prompt_y

text 임베딩은 frozen LLM 것을 data/text_encoder 가 도메인별로 캐시한다.
trainer 는 is_multimodal 플래그로 forward(x_enc, text_emb) 를 호출한다.
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


class MMFusion(nn.Module):
    is_multimodal = True

    def __init__(self, configs: ModelConfig, backbone="PatchTST"):
        super().__init__()
        self.backbone = _BACKBONES[backbone](configs)
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.prompt_weight = configs.prompt_weight
        # 원본 MLP: d_llm -> d_llm/8 -> pred_len*c_out (text_emb=pred_len)
        hidden = max(configs.d_llm // 8, 1)
        self.text_head = nn.Sequential(
            nn.Linear(configs.d_llm, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, configs.pred_len * configs.c_out),
        )

    def forward(self, x_enc, text_emb):
        ts = self.backbone(x_enc, None, None, None)                       # [B, H, V]
        prior = x_enc.mean(1, keepdim=True).expand(-1, self.pred_len, -1)  # 과거평균 레벨
        txt = self.text_head(text_emb).view(x_enc.size(0), self.pred_len, self.c_out)
        # 텍스트는 시퀀스 모양만 예측 (pred_len 축 정규화), 레벨은 prior 가 담당
        txt = txt - txt.mean(1, keepdim=True).detach()
        txt = txt / torch.sqrt(txt.var(1, keepdim=True, unbiased=False) + 1e-5)
        return (1 - self.prompt_weight) * ts + self.prompt_weight * (txt + prior)
