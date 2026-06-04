r"""튜닝된 베이스라인 표 — MM 행을 모델별 최적 prompt_weight(w*)로 교체.

입력:
  results/optimal_weights.csv                 (백본별 w*, val 선택)
  results/indomain_20260604_070005.csv        (unimodal full run; MM@0.1 포함하나 unimodal만 사용)
  results/sweep_weight_*.csv                  (weight별 결과 → MM@w* 추출)
산출:
  results/indomain_tuned_<stamp>.csv          (unimodal + MM@w*, seed별 raw; aggregate/compare 재사용 가능)
  results/indomain_tuned_<stamp>_seedavg.csv  (3-seed 평균)
  results/tables_mse_tuned.tex                (도메인별 MSE 표, MM 행에 w* 표기, 더 좋은 쪽 bold)
필요 패키지: \usepackage{booktabs}
실행: uv run python -m src.make_tuned_baseline
"""
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BACKBONES = ["DLinear", "PatchTST", "iTransformer",
             "Transformer", "Autoformer", "Informer", "FEDformer"]
METRICS = ["mse", "mae", "rmse", "mape"]
UNI_CSV = RESULTS_DIR / "indomain_20260604_070005.csv"
KEEP = ["model", "domain", "horizon", "seed"] + METRICS


def _fmt(v):
    return f"{v:.3f}"


def main():
    wstar = pd.read_csv(RESULTS_DIR / "optimal_weights.csv").set_index("backbone").optimal_weight.to_dict()
    full = pd.read_csv(UNI_CSV)
    sweep = pd.read_csv(sorted(glob.glob(str(RESULTS_DIR / "sweep_weight_*.csv")))[-1])

    # unimodal 행 (full run에서) + MM@w* 행 (sweep에서)
    parts = [full[full.model.isin(BACKBONES)][KEEP]]
    for b in BACKBONES:
        w = wstar[b]
        sub = sweep[(sweep.model == f"MM-TSFlib-{b}") & (np.isclose(sweep.weight, w))]
        parts.append(sub[KEEP])
    tuned = pd.concat(parts, ignore_index=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = RESULTS_DIR / f"indomain_tuned_{stamp}.csv"
    tuned.to_csv(raw_path, index=False)

    avg = tuned.groupby(["model", "domain", "horizon"])[METRICS].mean().round(6).reset_index()
    avg_path = RESULTS_DIR / f"indomain_tuned_{stamp}_seedavg.csv"
    avg.to_csv(avg_path, index=False)
    print(f"[saved] {raw_path.name}  ({len(tuned)} rows)")
    print(f"[saved] {avg_path.name}  ({len(avg)} rows, 3-seed mean)")

    # ── per-domain MSE LaTeX (MM 행에 w* 표기) ──
    domains = list(dict.fromkeys(avg.domain))
    horizons = sorted(avg.horizon.unique())
    blocks = [
        f"% Auto-generated tuned baseline from {raw_path.name}. Requires: \\usepackage{{booktabs}}",
        "% Uni = unimodal backbone; MM = MM-TSFlib fusion at per-model val-selected prompt_weight (w*).",
        "",
    ]
    for d in domains:
        L = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{In-domain MSE on \textsc{{{d}}} ($L{{=}}8$, 3-seed mean). "
             rf"\textbf{{Bold}} = better of Uni/MM.}}",
             rf"\label{{tab:tuned-{d.lower()}}}",
             r"\begin{tabular}{ll" + "c" * len(horizons) + "}",
             r"\toprule",
             "Model & Modal & " + " & ".join(rf"$H{{=}}{h}$" for h in horizons) + r" \\",
             r"\midrule"]
        for bi, b in enumerate(BACKBONES):
            u = avg[(avg.model == b) & (avg.domain == d)].set_index("horizon").mse
            m = avg[(avg.model == f"MM-TSFlib-{b}") & (avg.domain == d)].set_index("horizon").mse
            uc, mc = [], []
            for h in horizons:
                us, ms = _fmt(u[h]), _fmt(m[h])
                if u[h] <= m[h]:
                    us = rf"\textbf{{{us}}}"
                else:
                    ms = rf"\textbf{{{ms}}}"
                uc.append(us); mc.append(ms)
            L.append(rf"{b} & Uni & " + " & ".join(uc) + r" \\")
            L.append(r"& MM & " + " & ".join(mc) + r" \\")
            if bi != len(BACKBONES) - 1:
                L.append(r"\addlinespace[2pt]")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        blocks.append("\n".join(L))

    tex_path = RESULTS_DIR / "tables_mse_tuned.tex"
    tex_path.write_text("\n".join(blocks))
    print(f"[saved] {tex_path.name}  ({len(domains)} domain tables)")


if __name__ == "__main__":
    main()
