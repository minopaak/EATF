"""예측 평가 metric.

기본은 RevIN 정규화 공간에서 계산 (도메인 간 스케일이 통일돼 cross-domain 비교 가능).
원 스케일이 필요하면 호출부에서 mean/std로 역정규화 후 넘기면 됨.

다변량인 경우 target 채널(OT)만 보고 (target_idx).
"""
import torch


def mse(pred: torch.Tensor, true: torch.Tensor) -> float:
    return torch.mean((pred - true) ** 2).item()


def mae(pred: torch.Tensor, true: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - true)).item()


def compute_metrics(preds: torch.Tensor, trues: torch.Tensor, target_idx=0) -> dict:
    """preds, trues: [N, H, V]. MSE/MAE 계산.

    target_idx=int  : 그 채널(보통 OT)만
    target_idx=None : 전 채널 평균 (MM-TSFlib features='M' 방식)
    """
    if target_idx is None:
        p, t = preds, trues
    else:
        p, t = preds[..., target_idx], trues[..., target_idx]
    return {"mse": mse(p, t), "mae": mae(p, t)}
