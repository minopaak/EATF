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
from .architectures import PatchTST, DLinear

_MODELS = {
    "PatchTST": PatchTST,   # transformer, patch 기반 (ICLR'23)
    "DLinear":  DLinear,    # 분해 + 선형 (AAAI'23)
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

    for name in available_models():
        model = build_model(name, cfg)
        model.eval()
        with torch.no_grad():
            out = model(x, None, None, None)
        n_params = sum(p.numel() for p in model.parameters())
        ok = tuple(out.shape) == (B, H, V)
        print(f"{name:10s} | out={tuple(out.shape)} | params={n_params:,} | shape_ok={ok}")
