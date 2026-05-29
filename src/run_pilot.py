"""TS-swap LODO 파일럿 (REFERENCE ONLY — 헤드라인 cross-domain 실험 아님).

비교:
  - in-domain   : Agriculture 학습 → Agriculture test
  - cross-domain: Economy 학습     → Agriculture test (TS-swap zero-shot LODO)

NOTE (2026-05): 단순 TS-swap LODO는 도메인 간 시계열 동역학이 근본적으로 달라
trivial fail이고 우리 contribution과 정렬되지 않음. 헤드라인 cross-domain은
multimodal 모델의 텍스트/이벤트 지식 transfer 테스트(Phase 5). 이 스크립트는
파이프라인 sanity check + TS-only reference 수치용. 자세한 설계는 04_evaluation.md.

실행: EATF Dataset/ 에서
    uv run python -m src.run_pilot
"""
import torch

from .data import build_in_domain, build_zero_shot_lodo
from .models import build_model, ModelConfig
from .training import train_model, predict
from .evaluation import compute_metrics

SOURCE, TARGET = "Economy", "Agriculture"
L, H = 8, 12
MODELS = ["DLinear", "PatchTST"]
SEED = 2024
DEVICE = "cpu"


def _run(model_name, splits, normalize):
    cfg = ModelConfig(seq_len=L, pred_len=H,
                      enc_in=len(splits["var_cols"]), c_out=len(splits["var_cols"]))
    model = build_model(model_name, cfg)
    model = train_model(model, splits["train"], splits["val"],
                        epochs=10, lr=1e-4, patience=5, lradj="type1", device=DEVICE, seed=SEED)
    preds, trues, _, _ = predict(model, splits["test"], device=DEVICE)
    return compute_metrics(preds, trues, target_idx=splits["target_idx"])


def _run_block(normalize):
    in_splits = build_in_domain(TARGET, L, H, normalize=normalize)            # Agri -> Agri
    cd_splits = build_zero_shot_lodo(SOURCE, TARGET, L, H, normalize=normalize)  # Econ -> Agri
    print(f"\n--- normalize='{normalize}' --- "
          f"(in train={len(in_splits['train'])} test={len(in_splits['test'])}, "
          f"cross train={len(cd_splits['train'])} test={len(cd_splits['test'])})")
    print(f"{'model':10s} {'in MSE':>9s} {'cross MSE':>10s} {'degr(x/in)':>11s} "
          f"{'in MAE':>8s} {'cross MAE':>10s}")
    for name in MODELS:
        m_in = _run(name, in_splits, normalize)
        m_cd = _run(name, cd_splits, normalize)
        degr = m_cd["mse"] / m_in["mse"] if m_in["mse"] > 0 else float("nan")
        print(f"{name:10s} {m_in['mse']:9.4f} {m_cd['mse']:10.4f} {degr:11.2f} "
              f"{m_in['mae']:8.4f} {m_cd['mae']:10.4f}")


def main():
    torch.manual_seed(SEED)
    print(f"=== Phase 2 Pilot ===  L={L}, H={H}, seed={SEED}")
    print(f"in-domain: {TARGET}->{TARGET}   cross: {SOURCE}->{TARGET}  (eval = {TARGET} test split)")

    # 기본(dataset=per-domain global) vs RevIN(instance) 비교
    for normalize in ("dataset", "instance"):
        _run_block(normalize)

    print("\n(MSE/MAE는 각 정규화 공간 기준. degradation>1 이면 cross-domain에서 성능 저하)")


if __name__ == "__main__":
    main()
