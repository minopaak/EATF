r"""prompt_weight 스윕 분석 -> 최적 weight 아티팩트 + LaTeX 표.

입력: 최신 results/sweep_weight_*.csv (weight별 val/test) + unimodal full run
      (results/indomain_20260604_070005.csv) — unimodal test 기준선.

산출:
  results/optimal_weights.csv     : 백본별 최적 prompt_weight (val 기준 선택) + weight별 test 이득
  results/table_weight_sweep.tex  : 행=백본, 열=weight, 값=텍스트 이득%(vs unimodal, test),
                                    val로 선택된 최적 weight를 \textbf 강조.
필요 패키지: \usepackage{booktabs}
실행: uv run python -m src.select_weight [unimodal_csv]
"""
import glob
import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BACKBONES = ["DLinear", "PatchTST", "iTransformer",
             "Transformer", "Autoformer", "Informer", "FEDformer"]


def main():
    sweep = pd.read_csv(sorted(glob.glob(str(RESULTS_DIR / "sweep_weight_*.csv")))[-1])
    uni_csv = sys.argv[1] if len(sys.argv) > 1 else str(RESULTS_DIR / "indomain_20260604_070005.csv")
    full = pd.read_csv(uni_csv)
    weights = sorted(sweep.weight.unique())

    art_rows, tex_rows = [], []
    for b in BACKBONES:
        mm = sweep[sweep.model == f"MM-TSFlib-{b}"]
        # 셀(domain,horizon) 단위 seed 평균
        cell = mm.groupby(["domain", "horizon", "weight"]).agg(
            val=("val_mse", "mean"), test=("mse", "mean")).reset_index()
        uni = full[full.model == b].groupby(["domain", "horizon"]).mse.mean()

        # val 기준 최적 weight (셀별 정규화 후 weight 평균 최소 — 스케일 무관)
        cell["valn"] = cell.groupby(["domain", "horizon"]).val.transform(lambda x: x / x.mean())
        w_star = float(cell.groupby("weight").valn.mean().idxmin())

        # weight별 텍스트 이득%(test, vs unimodal) — 셀별 상대개선 평균
        gains = {}
        for w in weights:
            mw = cell[cell.weight == w].set_index(["domain", "horizon"]).test
            gains[w] = float(((uni - mw) / uni * 100).mean())
        art_rows.append({"backbone": b, "optimal_weight": w_star,
                         **{f"gain@{w}": round(gains[w], 3) for w in weights}})

        # LaTeX 행: 최적 weight 셀 bold
        cells = []
        for w in weights:
            s = f"{gains[w]:+.2f}"
            cells.append(rf"\textbf{{{s}}}" if w == w_star else s)
        tex_rows.append(f"{b} & " + " & ".join(cells) + r" \\")

    # 아티팩트 CSV
    art = pd.DataFrame(art_rows)
    art.to_csv(RESULTS_DIR / "optimal_weights.csv", index=False)
    print("[saved] results/optimal_weights.csv")
    print(art.to_string(index=False))

    # LaTeX 표
    tex = [
        "% Auto-generated weight-sweep table. Requires: \\usepackage{booktabs}",
        "% Value = text gain % (unimodal MSE - MM MSE)/unimodal, test, mean over 5 domains x 4 horizons.",
        "% Bold = val-selected optimal prompt_weight per backbone.",
        r"\begin{table}[t]", r"\centering",
        r"\caption{Text-fusion gain (\%) over the unimodal backbone as a function of "
        r"\texttt{prompt\_weight} (in-domain test MSE, mean over 5 domains $\times$ 4 horizons, "
        r"3-seed). \textbf{Bold} marks the val-selected optimum.}",
        r"\label{tab:weight-sweep}",
        r"\begin{tabular}{l" + "c" * len(weights) + "}",
        r"\toprule",
        "Backbone & " + " & ".join(rf"$w{{=}}{w}$" for w in weights) + r" \\",
        r"\midrule",
        *tex_rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]
    (RESULTS_DIR / "table_weight_sweep.tex").write_text("\n".join(tex))
    print("\n[saved] results/table_weight_sweep.tex")


if __name__ == "__main__":
    main()
