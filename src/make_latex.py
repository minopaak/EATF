r"""seed 평균 결과 -> Overleaf용 LaTeX 표 (도메인별, paper 스타일).

results/ 의 최신 *_seedavg.csv 를 읽어 도메인별로 표 하나씩 생성:
  행 = 백본 × {Uni(unimodal), MM(MM-TSFlib fusion)}, 열 = horizon, 값 = 지표(기본 MSE).
각 (도메인, horizon) 셀에서 Uni/MM 중 더 좋은(낮은) 값을 \textbf 로 강조.

필요 패키지(Overleaf preamble): \usepackage{booktabs}
실행: EATF Dataset/ 에서
    uv run python -m src.make_latex                 # 기본 metric=mse
    uv run python -m src.make_latex --metric mae
"""
import argparse
import glob
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BACKBONES = ["DLinear", "PatchTST", "iTransformer",
             "Transformer", "Autoformer", "Informer", "FEDformer"]


def _fmt(v):
    return f"{v:.3f}"


def _domain_table(df, domain, horizons, metric):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{In-domain {metric.upper()} on \textsc{{{domain}}} "
                 rf"($L{{=}}8$, 3-seed mean). \textbf{{Bold}} = better of Uni/MM.}}")
    lines.append(rf"\label{{tab:indomain-{domain.lower()}-{metric}}}")
    lines.append(r"\begin{tabular}{ll" + "c" * len(horizons) + "}")
    lines.append(r"\toprule")
    lines.append("Model & Modal & " + " & ".join(rf"$H{{=}}{h}$" for h in horizons) + r" \\")
    lines.append(r"\midrule")
    for bi, b in enumerate(BACKBONES):
        u = df[(df.model == b) & (df.domain == domain)].set_index("horizon")[metric]
        m = df[(df.model == f"MM-TSFlib-{b}") & (df.domain == domain)].set_index("horizon")[metric]
        ucell, mcell = [], []
        for h in horizons:
            uv, mv = u[h], m[h]
            us, ms = _fmt(uv), _fmt(mv)
            if uv <= mv:
                us = rf"\textbf{{{us}}}"
            else:
                ms = rf"\textbf{{{ms}}}"
            ucell.append(us)
            mcell.append(ms)
        lines.append(rf"{b} & Uni & " + " & ".join(ucell) + r" \\")
        lines.append(r"& MM & " + " & ".join(mcell) + r" \\")
        if bi != len(BACKBONES) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="mse", choices=["mse", "mae", "rmse", "mape"])
    args = ap.parse_args()
    metric = args.metric

    files = sorted(glob.glob(str(RESULTS_DIR / "*_seedavg.csv")))
    if not files:
        raise FileNotFoundError("*_seedavg.csv 없음. 먼저 src.aggregate 실행 필요.")
    src = Path(files[-1])
    df = pd.read_csv(src)
    domains = list(dict.fromkeys(df.domain))
    horizons = sorted(df.horizon.unique())

    blocks = [
        "% Auto-generated from " + src.name,
        "% Requires: \\usepackage{booktabs}",
        f"% Metric: {metric.upper()} (per-domain OT, 3-seed mean)",
        "",
    ]
    blocks += [_domain_table(df, d, horizons, metric) + "\n" for d in domains]

    dst = RESULTS_DIR / f"tables_{metric}.tex"
    dst.write_text("\n".join(blocks))
    print(f"[saved] {dst}")
    print(f"  {len(domains)}개 도메인 표 (metric={metric})")


if __name__ == "__main__":
    main()
