"""실험 결과 후처리 — seed 평균 CSV / unimodal-vs-MM 비교 / LaTeX 표.

results/ 의 최신 indomain_*.csv (run.py 출력, seed별 raw)를 읽어 처리한다.

실행: EATF Dataset/ 에서
    uv run python -m src.report seedavg              # 3-seed 평균+std CSV 저장
    uv run python -m src.report compare --metric mse # uni vs MM 나란히 출력
    uv run python -m src.report latex   --metric mse # 도메인별 LaTeX 표 (booktabs)
"""
import argparse
import glob
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BACKBONES = ["DLinear", "PatchTST", "iTransformer",
             "Transformer", "Autoformer", "Informer", "FEDformer"]
METRICS = ["mse", "mae", "rmse", "mape"]


def _load():
    files = [f for f in sorted(glob.glob(str(RESULTS_DIR / "indomain_*.csv")))
             if not f.endswith("_seedavg.csv")]
    if not files:
        raise FileNotFoundError("indomain_*.csv 없음. 먼저 src.run 실행 필요.")
    return pd.read_csv(files[-1]), Path(files[-1])


def seedavg():
    df, src = _load()
    g = df.groupby(["model", "domain", "horizon"])[METRICS]
    std = g.std(ddof=0).round(6)
    std.columns = [f"{m}_std" for m in METRICS]
    out = pd.concat([g.mean().round(6), std], axis=1).reset_index()
    dst = src.with_name(src.stem + "_seedavg.csv")
    out.to_csv(dst, index=False)
    print(f"[saved] {dst}  ({len(out)} rows, 3-seed 평균+std)")


def compare(metric):
    df, _ = _load()
    horizons = sorted(df.horizon.unique())
    domains = list(dict.fromkeys(df.domain))
    print(f"=== Unimodal vs Multimodal (OT {metric.upper()}, 3-seed 평균) ===\n")
    for b in BACKBONES:
        u = df[df.model == b].groupby(["domain", "horizon"])[metric].mean()
        m = df[df.model == f"MM-TSFlib-{b}"].groupby(["domain", "horizon"])[metric].mean()
        print(f"#####  {b}  vs  MM-TSFlib-{b}  #####")
        deltas = []
        for d in domains:
            cells = []
            for h in horizons:
                dp = (m[d, h] - u[d, h]) / u[d, h] * 100
                deltas.append(dp)
                cells.append(f"{u[d, h]:.3f}->{m[d, h]:.3f}({dp:+.1f})")
            print(f"{d:12s} " + "  ".join(cells))
        win = sum(x < 0 for x in deltas)
        print(f"  평균 Δ%={sum(deltas) / len(deltas):+.2f}, 개선 {win}/{len(deltas)} 셀\n")


def latex(metric):
    df, _ = _load()
    avg = df.groupby(["model", "domain", "horizon"])[metric].mean()
    horizons = sorted(df.horizon.unique())
    domains = list(dict.fromkeys(df.domain))
    out = ["% Requires: \\usepackage{booktabs}", ""]
    for d in domains:
        out += [r"\begin{table}[t]", r"\centering",
                rf"\caption{{In-domain {metric.upper()} on \textsc{{{d}}} "
                rf"($L{{=}}8$, 3-seed mean). \textbf{{Bold}} = better of Uni/MM.}}",
                rf"\label{{tab:indomain-{d.lower()}-{metric}}}",
                r"\begin{tabular}{ll" + "c" * len(horizons) + "}", r"\toprule",
                "Model & Modal & " + " & ".join(rf"$H{{=}}{h}$" for h in horizons) + r" \\",
                r"\midrule"]
        for bi, b in enumerate(BACKBONES):
            uc, mc = [], []
            for h in horizons:
                uv, mv = avg[b, d, h], avg[f"MM-TSFlib-{b}", d, h]
                us, ms = f"{uv:.3f}", f"{mv:.3f}"
                if uv <= mv:
                    us = rf"\textbf{{{us}}}"
                else:
                    ms = rf"\textbf{{{ms}}}"
                uc.append(us)
                mc.append(ms)
            out.append(rf"{b} & Uni & " + " & ".join(uc) + r" \\")
            out.append(r"& MM & " + " & ".join(mc) + r" \\")
            if bi != len(BACKBONES) - 1:
                out.append(r"\addlinespace[2pt]")
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    dst = RESULTS_DIR / f"tables_{metric}.tex"
    dst.write_text("\n".join(out))
    print(f"[saved] {dst}  ({len(domains)} domain tables)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seedavg")
    for name in ("compare", "latex"):
        p = sub.add_parser(name)
        p.add_argument("--metric", default="mse", choices=METRICS)
    args = ap.parse_args()
    if args.cmd == "seedavg":
        seedavg()
    elif args.cmd == "compare":
        compare(args.metric)
    else:
        latex(args.metric)


if __name__ == "__main__":
    main()
