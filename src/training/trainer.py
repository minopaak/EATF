"""학습 / 예측 루프.

- 손실은 RevIN 정규화 공간에서 MSE (Dataset이 정규화된 x,y를 줌).
- val이 있으면 early stopping (patience), 없으면 고정 epoch.
- 모델 forward 규약: model(x, None, None, None) -> [B, H, V]
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _forward(model, x, te, device):
    """모델 forward. is_multimodal이면 텍스트 임베딩(te)을 함께 넘긴다."""
    if getattr(model, "is_multimodal", False):
        return model(x.to(device), te.to(device))
    return model(x.to(device), None, None, None)


def _eval_loss(model, loader, crit, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, y, _, _, te in loader:
            y = y.to(device)
            pred = _forward(model, x, te, device)
            tot += crit(pred, y).item() * len(x)
            n += len(x)
    return tot / max(n, 1)


def train_model(model, train_ds, val_ds=None, *, epochs=10, lr=1e-4, batch_size=32,
                patience=5, lradj="type1", device="cpu", seed=0, verbose=False):
    """train_ds/val_ds: WindowDataset(또는 ConcatDataset). 최적 val 가중치로 복원해 반환.

    MM-TSFlib/TSLib 표준 설정: lr=1e-4, epochs=10, patience=5, lradj='type1'(매 epoch lr 절반).
    """
    torch.manual_seed(seed)
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    has_val = val_ds is not None and len(val_ds) > 0
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if has_val else None

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        # TSLib type1: 매 epoch lr 절반 감쇠 (epoch1=lr, epoch2=lr/2, ...)
        if lradj == "type1":
            cur_lr = lr * (0.5 ** ep)
            for g in opt.param_groups:
                g["lr"] = cur_lr

        model.train()
        for x, y, _, _, te in train_loader:
            y = y.to(device)
            opt.zero_grad()
            loss = crit(_forward(model, x, te, device), y)
            loss.backward()
            opt.step()

        if val_loader is not None:
            vl = _eval_loss(model, val_loader, crit, device)
            if verbose:
                print(f"    epoch {ep+1:02d} val_loss={vl:.4f}")
            if vl < best_val - 1e-6:
                best_val, best_state, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"    early stop @ epoch {ep+1}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, ds, *, batch_size=64, device="cpu"):
    """ds 전체에 대해 (preds, trues, means, stds) 반환. 모두 [N, H, V] / [N, V]."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    preds, trues, means, stds = [], [], [], []
    for x, y, m, s, te in loader:
        p = _forward(model, x, te, device).cpu()
        preds.append(p)
        trues.append(y)
        means.append(m)
        stds.append(s)
    return torch.cat(preds), torch.cat(trues), torch.cat(means), torch.cat(stds)
