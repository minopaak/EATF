"""seed별 raw 결과 CSV -> seed 평균 CSV.

results/ 의 최신 indomain_*.csv (seed별 raw)를 읽어 (model, domain, horizon) 단위로
4지표(mse/mae/rmse/mape)의 3-seed 평균과 표준편차를 계산해 *_seedavg.csv 로 저장.

실행: EATF Dataset/ 에서
    uv run python -m src.aggregate
    uv run python -m src.aggregate results/indomain_20260604_070005.csv
"""
import glob
import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
METRICS = ["mse", "mae", "rmse", "mape"]


def main():
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
    else:
        files = sorted(glob.glob(str(RESULTS_DIR / "indomain_*.csv")))
        files = [f for f in files if not f.endswith("_seedavg.csv")]
        if not files:
            raise FileNotFoundError("indomain_*.csv 없음. 먼저 src.run 실행 필요.")
        src = Path(files[-1])

    df = pd.read_csv(src)
    g = df.groupby(["model", "domain", "horizon"])[METRICS]
    agg = g.mean().round(6)
    agg.columns = METRICS                                  # 평균 (지표명 그대로)
    std = g.std(ddof=0).round(6)
    std.columns = [f"{m}_std" for m in METRICS]            # seed 표준편차
    n = g.size().rename("n_seeds")
    out = pd.concat([agg, std, n], axis=1).reset_index()
    # 컬럼 순서: 키 + 지표평균 + 지표std + n
    out = out[["model", "domain", "horizon"] + METRICS
              + [f"{m}_std" for m in METRICS] + ["n_seeds"]]

    dst = src.with_name(src.stem + "_seedavg.csv")
    out.to_csv(dst, index=False)
    print(f"[saved] {dst}")
    print(f"  {len(out)} rows = {out.model.nunique()}모델 × "
          f"{out.domain.nunique()}도메인 × {out.horizon.nunique()}horizon (3-seed 평균)")


if __name__ == "__main__":
    main()
