"""모델 빌더 / 레지스트리.

우리 프로젝트 안에 직접 구현한 모델들을 이름으로 생성한다.
(clones/Time-Series-Library 의 forecasting 코드를 self-contained하게 옮겨온
src/models/*.py 를 사용 — 외부 import 의존 없음.)

사용:
    from src.models import build_model, ModelConfig
    cfg = ModelConfig(seq_len=36, pred_len=12, enc_in=1)
    model = build_model("PatchTST", cfg)
    # forward: (x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None) -> [B, pred_len, D]
"""
import torch.nn as nn

from .config import ModelConfig
from .architectures import (PatchTST, DLinear, iTransformer,
                            Transformer, Autoformer, Informer, FEDformer, MMFusion)

_MODELS = {
    # ── unimodal ──────────────────────────────────────────
    "PatchTST":     PatchTST,      # transformer, patch 기반 (ICLR'23)
    "DLinear":      DLinear,       # 분해 + 선형 (AAAI'23)
    "iTransformer": iTransformer,  # 변수-토큰 inverted attention (ICLR'24)
    "Transformer":  Transformer,   # vanilla enc-dec (NeurIPS'17)
    "Autoformer":   Autoformer,    # AutoCorrelation + decomp (NeurIPS'21)
    "Informer":     Informer,      # ProbSparse attention (AAAI'21)
    "FEDformer":    FEDformer,     # Fourier enhanced decomp (ICML'22)
    # ── multimodal (MM-TSFlib 계열 fusion; TS 백본 + 텍스트 헤드) ──
    "MM-TSFlib-PatchTST":     lambda c: MMFusion(c, "PatchTST"),
    "MM-TSFlib-DLinear":      lambda c: MMFusion(c, "DLinear"),
    "MM-TSFlib-iTransformer": lambda c: MMFusion(c, "iTransformer"),
    "MM-TSFlib-Transformer":  lambda c: MMFusion(c, "Transformer"),
    "MM-TSFlib-Autoformer":   lambda c: MMFusion(c, "Autoformer"),
    "MM-TSFlib-Informer":     lambda c: MMFusion(c, "Informer"),
    "MM-TSFlib-FEDformer":    lambda c: MMFusion(c, "FEDformer"),
}


def available_models() -> list:
    return list(_MODELS.keys())


def build_model(name: str, config: ModelConfig) -> nn.Module:
    """이름으로 모델 인스턴스 생성."""
    if name not in _MODELS:
        raise ValueError(f"Unknown model: {name!r}. Available: {available_models()}")
    return _MODELS[name](config)


if __name__ == "__main__":
    # smoke test: 모델 빌드 + 더미 forward로 출력 shape 확인
    import torch

    B, L, H, V = 4, 36, 12, 3
    cfg = ModelConfig(seq_len=L, pred_len=H, enc_in=V, c_out=V)
    x = torch.randn(B, L, V)

    te = torch.randn(B, cfg.d_llm)   # 멀티모달용 더미 텍스트 임베딩
    for name in available_models():
        model = build_model(name, cfg)
        model.eval()
        with torch.no_grad():
            if getattr(model, "is_multimodal", False):
                out = model(x, te)
            else:
                out = model(x, None, None, None)
        n_params = sum(p.numel() for p in model.parameters())
        ok = tuple(out.shape) == (B, H, V)
        print(f"{name:20s} | out={tuple(out.shape)} | params={n_params:,} | shape_ok={ok}")
