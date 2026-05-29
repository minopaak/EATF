"""In-domain 베이스라인 (3 seed 평균) — Time-MMD monthly 설정.

입력은 다변량(multivariate=True): 다변량 도메인(Agriculture, Economy)은 전 변수를
입력 채널로 사용(공통 윈도우), 단변량 도메인(Security/SocialGood/Traffic)은 OT 1채널.
평가는 모든 도메인 OT 채널 MSE(target_idx=OT)로 통일 — cross-domain LODO와 일관.
L=8, horizons {6,8,10,12}, dataset(per-domain global) 표준화,
학습 프로토콜 MM-TSFlib 정렬(lr=1e-4, type1, epochs=10, patience=5).

실행: EATF Dataset/ 에서
    uv run python -m src.run_indomain
"""
from statistics import mean, stdev

from .data import build_in_domain, MONTHLY_DOMAINS
from .models import build_model, ModelConfig
from .training import train_model, predict
from .evaluation import compute_metrics

L = 8
HORIZONS = [6, 8, 10, 12]
MODELS = ["DLinear", "PatchTST"]
SEEDS = [2024, 2025, 2026]
DEVICE = "cpu"


def _seed_avg(mname, sp, H):
    nvar = len(sp["var_cols"])
    tidx = sp["target_idx"]
    vals = []
    for seed in SEEDS:
        cfg = ModelConfig(seq_len=L, pred_len=H, enc_in=nvar, c_out=nvar)
        model = build_model(mname, cfg)
        model = train_model(model, sp["train"], sp["val"], epochs=10, lr=1e-4,
                            patience=5, lradj="type1", device=DEVICE, seed=seed)
        p, t, _, _ = predict(model, sp["test"], device=DEVICE)
        vals.append(compute_metrics(p, t, target_idx=tidx)["mse"])
    return mean(vals)


def main():
    print(f"=== In-domain Baseline (다변량 입력·OT 평가, 3-seed 평균) ===")
    print(f"L={L}, horizons={HORIZONS}, normalize=dataset, lr=1e-4/type1/10ep\n")

    # (H, domain) -> splits (다변량 입력: 다변량 도메인=전 변수/공통윈도우, 단변량=OT/풀)
    splits = {(H, d): build_in_domain(d, L, H, multivariate=True)
              for H in HORIZONS for d in MONTHLY_DOMAINS}

    # 도메인별 실제 입력 변수 표시
    nvar0 = {d: len(splits[(HORIZONS[0], d)]["var_cols"]) for d in MONTHLY_DOMAINS}
    print("입력 변수 수: " + ", ".join(f"{d}={nvar0[d]}" for d in MONTHLY_DOMAINS) + "\n")

    for mname in MODELS:
        print(f"\n#################  {mname}  (OT MSE) #################")
        print(f"{'domain':12s} " + "".join(f"{'H='+str(H):>9s}" for H in HORIZONS))
        for d in MONTHLY_DOMAINS:
            cells = [_seed_avg(mname, splits[(H, d)], H) for H in HORIZONS]
            print(f"{d:12s} " + "".join(f"{c:9.3f}" for c in cells))

    print("\n(per-domain 표준화 공간 OT MSE, 3-seed 평균. test=각 도메인 마지막 20%)")
    print("(다변량 도메인은 전 변수 입력이나 평가는 OT 채널만)")


if __name__ == "__main__":
    main()
