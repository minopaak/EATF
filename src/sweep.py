r"""prompt_weight 스윕 (ablation) — 실행 + 분석.

run    : MM 모델 × 도메인 × horizon × weight × seed 학습 → val/test 기록.
         (누수 방지: weight 선택은 val, 보고는 test)
analyze: 백본별 최적 weight(val 기준) + weight별 텍스트 이득표(LaTeX) 산출.

실행: EATF Dataset/ 에서
    CUDA_VISIBLE_DEVICES=4 uv run python -m src.sweep run
    uv run python -m src.sweep analyze
"""
import argparse
import csv
import glob
from datetime import datetime
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BACKBONES = ["DLinear", "PatchTST", "iTransformer",
             "Transformer", "Autoformer", "Informer", "FEDformer"]
METRICS = ["mse", "mae", "rmse", "mape"]
L, HORIZONS, SEEDS = 8, [6, 8, 10, 12], [2024, 2025, 2026]
WEIGHTS = [0.05, 0.1, 0.2, 0.3, 0.5]
LLM, D_LLM = "BERT", 768


def run():
    import torch
    from .data import build_in_domain, MONTHLY_DOMAINS
    from .models import build_model, ModelConfig
    from .training import train_model, predict
    from .evaluation import compute_metrics

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = {(H, d): build_in_domain(d, L, H, multivariate=True, text=True, device=device)
              for H in HORIZONS for d in MONTHLY_DOMAINS}
    records = []
    for b in BACKBONES:
        mname = f"MM-TSFlib-{b}"
        for d in MONTHLY_DOMAINS:
            for H in HORIZONS:
                sp = splits[(H, d)]
                nvar, tidx = len(sp["var_cols"]), sp["target_idx"]
                for w in WEIGHTS:
                    for seed in SEEDS:
                        cfg = ModelConfig(seq_len=L, pred_len=H, enc_in=nvar, c_out=nvar,
                                          use_text=True, llm_model=LLM, d_llm=D_LLM, prompt_weight=w)
                        model = train_model(build_model(mname, cfg), sp["train"], sp["val"],
                                            epochs=10, lr=1e-4, patience=5, lradj="type1",
                                            device=device, seed=seed)
                        vp, vt, _, _ = predict(model, sp["val"], device=device)
                        p, t, m, s = predict(model, sp["test"], device=device)
                        records.append({"model": mname, "domain": d, "horizon": H,
                                        "weight": w, "seed": seed,
                                        "val_mse": round(compute_metrics(vp, vt, target_idx=tidx)["mse"], 6),
                                        **compute_metrics(p, t, target_idx=tidx, mean=m, std=s)})
                print(f"  {mname} / {d} done")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"sweep_weight_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["model", "domain", "horizon", "weight", "seed", "val_mse"] + METRICS)
        wtr.writeheader()
        wtr.writerows(records)
    print(f"\n[saved] {out}  ({len(records)} rows)")


def analyze():
    sweep = pd.read_csv(sorted(glob.glob(str(RESULTS_DIR / "sweep_weight_*.csv")))[-1])
    uni_files = [f for f in sorted(glob.glob(str(RESULTS_DIR / "indomain_*.csv")))
                 if not f.endswith("_seedavg.csv")]
    full = pd.read_csv(uni_files[-1])
    weights = sorted(sweep.weight.unique())

    art, tex_rows = [], []
    for b in BACKBONES:
        cell = sweep[sweep.model == f"MM-TSFlib-{b}"].groupby(
            ["domain", "horizon", "weight"]).agg(val=("val_mse", "mean"), test=("mse", "mean")).reset_index()
        uni = full[full.model == b].groupby(["domain", "horizon"]).mse.mean()
        # val 기준 최적 weight (셀별 정규화 후 평균 최소 — 스케일 무관)
        cell["valn"] = cell.groupby(["domain", "horizon"]).val.transform(lambda x: x / x.mean())
        w_star = float(cell.groupby("weight").valn.mean().idxmin())
        gains = {w: float(((uni - cell[cell.weight == w].set_index(["domain", "horizon"]).test) / uni * 100).mean())
                 for w in weights}
        art.append({"backbone": b, "optimal_weight": w_star, **{f"gain@{w}": round(gains[w], 3) for w in weights}})
        cells = [rf"\textbf{{{gains[w]:+.2f}}}" if w == w_star else f"{gains[w]:+.2f}" for w in weights]
        tex_rows.append(f"{b} & " + " & ".join(cells) + r" \\")

    pd.DataFrame(art).to_csv(RESULTS_DIR / "optimal_weights.csv", index=False)
    print("[saved] results/optimal_weights.csv")
    print(pd.DataFrame(art).to_string(index=False))

    tex = ["% Requires: \\usepackage{booktabs}. Value = text gain % (test, vs unimodal).",
           r"\begin{table}[t]", r"\centering",
           r"\caption{Text-fusion gain (\%) over the unimodal backbone vs.\ \texttt{prompt\_weight} "
           r"(in-domain test MSE, mean over 5 domains $\times$ 4 horizons, 3-seed). "
           r"\textbf{Bold} = val-selected optimum.}",
           r"\label{tab:weight-sweep}",
           r"\begin{tabular}{l" + "c" * len(weights) + "}", r"\toprule",
           "Backbone & " + " & ".join(rf"$w{{=}}{w}$" for w in weights) + r" \\", r"\midrule",
           *tex_rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (RESULTS_DIR / "table_weight_sweep.tex").write_text("\n".join(tex))
    print("[saved] results/table_weight_sweep.tex")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("analyze")
    args = ap.parse_args()
    {"run": run, "analyze": analyze}[args.cmd]()


if __name__ == "__main__":
    main()
