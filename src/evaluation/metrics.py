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


def rmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - true) ** 2)).item()


def mape(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-3) -> float:
    """Mean Absolute Percentage Error (%). 분모 0 가드(eps).

    표준화 공간에서는 true가 0 근처를 지나 폭발하므로, 의미 있으려면
    원 스케일로 역정규화한 pred/true를 넘겨야 한다 (compute_metrics가 처리).
    """
    return torch.mean(torch.abs((pred - true) / (true.abs() + eps))).item() * 100.0


def compute_metrics(preds: torch.Tensor, trues: torch.Tensor, target_idx=0,
                    mean: torch.Tensor = None, std: torch.Tensor = None) -> dict:
    """preds, trues: [N, H, V]. MSE/MAE/RMSE/MAPE 계산.

    target_idx=int  : 그 채널(보통 OT)만
    target_idx=None : 전 채널 평균 (MM-TSFlib features='M' 방식)

    mean/std (predict()가 반환하는 [N, V] 스케일러)를 주면 MAPE는 **원 스케일로
    역정규화 후** 계산한다. MSE/MAE/RMSE는 표준화 공간 유지(cross-domain 비교용).
    안 주면 MAPE도 표준화 공간에서 계산(0-나눗셈으로 신뢰 불가).
    """
    if target_idx is None:
        p, t, m, s = preds, trues, mean, std
    else:
        p, t = preds[..., target_idx], trues[..., target_idx]
        m = mean[..., target_idx] if mean is not None else None
        s = std[..., target_idx] if std is not None else None

    out = {"mse": mse(p, t), "mae": mae(p, t), "rmse": rmse(p, t)}
    if m is not None and s is not None:
        m, s = m.unsqueeze(1), s.unsqueeze(1)   # [N] -> [N,1] (H축 broadcast)
        out["mape"] = mape(p * s + m, t * s + m)
    else:
        out["mape"] = mape(p, t)
    return out
