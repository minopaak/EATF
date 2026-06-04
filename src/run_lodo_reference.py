"""TS-swap LODO reference (REFERENCE ONLY — 헤드라인 cross-domain 아님).

각 source 도메인에 모델을 한 번만 학습 → 모든 target의 test split에 평가한 full 매트릭스.
  - 대각선 (source==target) = in-domain
  - 비대각선                 = cross-domain (TS-swap zero-shot LODO)
  - degradation[s, t] = MSE(s→t) / MSE(t→t)   (target의 in-domain 대비)
전부 OT 단변량, per-domain global 표준화(dataset). 평가 공간은 각 target의 표준화 공간.
(예: Economy→Agriculture 한 쌍만 보던 옛 pilot 은 이 매트릭스의 한 셀이라 여기에 포섭됨.)

NOTE (2026-05): TS-swap LODO는 도메인 간 시계열 동역학 차이로 trivial fail이라
헤드라인 결과 아님 (단순 reference). 헤드라인 cross-domain은 multimodal 모델의
텍스트/이벤트 지식 transfer 테스트(Phase 5). 04_evaluation.md 참조.

실행: EATF Dataset/ 에서
    uv run python -m src.run_lodo_reference
"""
import torch

from .data import build_in_domain, MONTHLY_DOMAINS
from .models import build_model, ModelConfig
from .training import train_model, predict
from .evaluation import compute_metrics

L, H = 8, 12
MODELS = ["DLinear", "PatchTST"]
SEED = 2024
DEVICE = "cpu"
ABBR = {"Agriculture": "Agri", "Economy": "Econ", "Security": "Sec",
        "SocialGood": "Social", "Traffic": "Traffic"}


def _eval(model, test_ds):
    preds, trues, _, _ = predict(model, test_ds, device=DEVICE)
    return compute_metrics(preds, trues, target_idx=0)["mse"]


def _print_matrix(title, mat, domains, fmt):
    print(title)
    print("src\\tgt  " + "".join(f"{ABBR[t]:>8s}" for t in domains))
    for s in domains:
        print(f"{ABBR[s]:8s}" + "".join(fmt(mat[(s, t)]) for t in domains))


def main():
    torch.manual_seed(SEED)
    domains = MONTHLY_DOMAINS
    print(f"=== TS-swap LODO Reference ===  L={L}, H={H}, seed={SEED}, normalize=dataset")
    print("(대각선=in-domain, 비대각선=cross-domain; REFERENCE ONLY)\n")

    splits = {d: build_in_domain(d, L, H) for d in domains}  # OT 단변량, dataset 표준화

    for mname in MODELS:
        print(f"\n########################  {mname}  ########################")
        cfg = ModelConfig(seq_len=L, pred_len=H, enc_in=1, c_out=1)

        # source별 1회 학습
        models = {}
        for s in domains:
            m = build_model(mname, cfg)
            models[s] = train_model(m, splits[s]["train"], splits[s]["val"],
                                    epochs=10, lr=1e-4, patience=5, lradj="type1",
                                    device=DEVICE, seed=SEED)

        # MSE 행렬 (source → target)
        mse = {(s, t): _eval(models[s], splits[t]["test"]) for s in domains for t in domains}

        _print_matrix("\n[MSE]  (행=source 학습, 열=target 평가)", mse, domains,
                      lambda v: f"{v:8.3f}")
        degr = {(s, t): (mse[(s, t)] / mse[(t, t)] if mse[(t, t)] > 0 else float("nan"))
                for s in domains for t in domains}
        _print_matrix("\n[Degradation]  MSE(s→t) / MSE(t→t)", degr, domains,
                      lambda v: f"{v:8.2f}")

        cross = [degr[(s, t)] for s in domains for t in domains if s != t]
        in_d = [mse[(t, t)] for t in domains]
        print(f"\n  평균 in-domain MSE      : {sum(in_d)/len(in_d):.3f}")
        print(f"  평균 cross degradation  : {sum(cross)/len(cross):.2f}  "
              f"(>1 이면 cross-domain 저하)")


if __name__ == "__main__":
    main()
