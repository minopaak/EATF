"""단일(unimodal) vs 멀티모달(MM-TSFlib) 베이스라인 seed 평균 비교.

results/ 의 최신 indomain_*.csv (run.py 통합 출력: unimodal + 멀티모달 한 파일)를
읽어, 백본별로 짝지어(DLinear↔MM-TSFlib-DLinear, PatchTST↔MM-TSFlib-PatchTST)
seed 평균값을 나란히 출력.

실행: EATF Dataset/ 에서
    uv run python -m src.compare              # 기본 metric=mse
    uv run python -m src.compare --metric mae
"""
import argparse
import glob
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BACKBONES = ["DLinear", "PatchTST", "iTransformer",
             "Transformer", "Autoformer", "Informer", "FEDformer"]
PAIRS = [(b, f"MM-TSFlib-{b}") for b in BACKBONES]


def _latest(pattern):
    files = sorted(glob.glob(str(RESULTS_DIR / pattern)))
    if not files:
        raise FileNotFoundError(f"{pattern} 없음. 먼저 베이스라인 실행 필요.")
    return files[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="mse", choices=["mse", "mae", "rmse", "mape"])
    args = ap.parse_args()
    metric = args.metric

    df = pd.read_csv(_latest("indomain_*.csv"))   # 통합 CSV (uni + mm 한 파일)
    uni = mm = df
    horizons = sorted(df.horizon.unique())
    domains = list(dict.fromkeys(df.domain))      # 입력 순서 보존

    print(f"=== Unimodal vs Multimodal (OT {metric.upper()}, 3-seed 평균) ===\n")
    for ub, mb in PAIRS:
        u = uni[uni.model == ub].groupby(["domain", "horizon"])[metric].mean()
        m = mm[mm.model == mb].groupby(["domain", "horizon"])[metric].mean()

        print(f"#####  {ub}  vs  {mb}  #####")
        head = "".join(f"{'H='+str(H):>16s}" for H in horizons)
        print(f"{'domain':12s} {head}")
        print(f"{'':12s} " + "".join(f"{'uni → mm  (Δ%)':>16s}" for _ in horizons))
        deltas = []
        for d in domains:
            cells = []
            for H in horizons:
                uv, mv = u[(d, H)], m[(d, H)]
                dp = (mv - uv) / uv * 100 if uv else 0.0
                deltas.append(dp)
                cells.append(f"{uv:.3f}→{mv:.3f}({dp:+.1f})")
            print(f"{d:12s} " + "".join(f"{c:>16s}" for c in cells))
        avg = sum(deltas) / len(deltas)
        win = sum(1 for x in deltas if x < 0)
        print(f"  → 평균 Δ%={avg:+.2f}%, 멀티모달 개선 {win}/{len(deltas)} 셀\n")


if __name__ == "__main__":
    main()
