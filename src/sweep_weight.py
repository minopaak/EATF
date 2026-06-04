"""prompt_weight 스윕 — MM-TSFlib 모델별 최적 텍스트 비중 탐색.

프로토콜(누수 방지): 각 (MM모델, 도메인, horizon, weight, seed)에 대해 학습 후
**val MSE 와 test 4지표**를 모두 기록한다. weight 선택은 val 로, 최종 보고는 test 로.
raw 를 저장하므로 per-model / per-(model,domain) 어느 granularity로도 사후 선택 가능.

출력:
  results/sweep_weight_<stamp>.csv   : model,domain,horizon,weight,seed,val_mse,mse,mae,rmse,mape
실행: EATF Dataset/ 에서
    CUDA_VISIBLE_DEVICES=4 uv run python -m src.sweep_weight
"""
import csv
from datetime import datetime
from pathlib import Path
from statistics import mean

import torch

from .data import build_in_domain, MONTHLY_DOMAINS
from .models import build_model, ModelConfig
from .training import train_model, predict
from .evaluation import compute_metrics

L = 8
HORIZONS = [6, 8, 10, 12]
SEEDS = [2024, 2025, 2026]
WEIGHTS = [0.05, 0.1, 0.2, 0.3, 0.5]
BACKBONES = ["DLinear", "PatchTST", "iTransformer",
             "Transformer", "Autoformer", "Informer", "FEDformer"]
MM_MODELS = [f"MM-TSFlib-{b}" for b in BACKBONES]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LLM, D_LLM, TEXT_SOURCE, TEXT_POOL = "BERT", 768, "both", "avg"
METRICS = ["mse", "mae", "rmse", "mape"]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main():
    print(f"=== prompt_weight sweep === device={DEVICE}")
    print(f"weights={WEIGHTS}, models={len(MM_MODELS)}, domains={len(MONTHLY_DOMAINS)}, "
          f"H={HORIZONS}, seeds={SEEDS}")
    total = len(MM_MODELS) * len(MONTHLY_DOMAINS) * len(HORIZONS) * len(WEIGHTS) * len(SEEDS)
    print(f"총 {total} run\n")

    # (H, domain) -> 텍스트 부착 split (도메인 임베딩은 캐시 재사용)
    splits = {(H, d): build_in_domain(d, L, H, multivariate=True, text=True,
                                      llm=LLM, text_source=TEXT_SOURCE,
                                      text_pool=TEXT_POOL, device=DEVICE)
              for H in HORIZONS for d in MONTHLY_DOMAINS}

    records = []
    done = 0
    for mname in MM_MODELS:
        for d in MONTHLY_DOMAINS:
            for H in HORIZONS:
                sp = splits[(H, d)]
                nvar = len(sp["var_cols"]); tidx = sp["target_idx"]
                for w in WEIGHTS:
                    for seed in SEEDS:
                        cfg = ModelConfig(seq_len=L, pred_len=H, enc_in=nvar, c_out=nvar,
                                          use_text=True, llm_model=LLM, d_llm=D_LLM,
                                          prompt_weight=w)
                        model = build_model(mname, cfg)
                        model = train_model(model, sp["train"], sp["val"], epochs=10,
                                             lr=1e-4, patience=5, lradj="type1",
                                             device=DEVICE, seed=seed)
                        vp, vt, vm, vs = predict(model, sp["val"], device=DEVICE)
                        val_mse = compute_metrics(vp, vt, target_idx=tidx)["mse"]
                        p, t, m, s = predict(model, sp["test"], device=DEVICE)
                        mt = compute_metrics(p, t, target_idx=tidx, mean=m, std=s)
                        records.append({"model": mname, "domain": d, "horizon": H,
                                        "weight": w, "seed": seed,
                                        "val_mse": round(val_mse, 6), **mt})
                        done += 1
                print(f"  [{done}/{total}] {mname} / {d} done")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"sweep_weight_{stamp}.csv"
    with open(out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["model", "domain", "horizon", "weight",
                                            "seed", "val_mse"] + METRICS)
        wtr.writeheader(); wtr.writerows(records)
    print(f"\n[saved] {out}  ({len(records)} rows)")


if __name__ == "__main__":
    main()
