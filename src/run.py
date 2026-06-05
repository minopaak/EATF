"""In-domain 베이스라인 통합 러너 (unimodal + 멀티모달, 3-seed 평균).

unimodal(DLinear/PatchTST)과 멀티모달(MM-TSFlib 계열)을 **하나의 모델 목록**으로 같은
train/val/test split 위에서 실행하고, 4지표(MSE/MAE/RMSE/MAPE)를 한 CSV로 저장한다.
목록에 멀티모달이 있으면 frozen LLM 텍스트 임베딩을 자동 부착(도메인별 1회 캐시);
unimodal 모델은 텍스트를 무시한다(trainer가 is_multimodal로 분기).

다변량 입력·OT 평가, L=8, H={6,8,10,12}, per-domain 표준화, lr=1e-4/type1/10ep.
MSE/MAE/RMSE=표준화 공간, MAPE=원 스케일 역정규화.

실행: EATF Dataset/ 에서
    uv run python -m src.run                                  # 기본 4모델
    uv run python -m src.run --models DLinear MM-TSFlib-DLinear
    uv run python -m src.run --device cpu
"""
import argparse
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
_BACKBONES = ["DLinear", "PatchTST", "iTransformer",
              "Transformer", "Autoformer", "Informer", "FEDformer"]
DEFAULT_MODELS = _BACKBONES + [f"MM-TSFlib-{b}" for b in _BACKBONES]   # unimodal 7 + MM 7

METRICS = ["mse", "mae", "rmse", "mape"]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# 텍스트 인코더 설정 (멀티모달 모델이 있을 때만 사용)
LLM = "BERT"
D_LLM = 768
TEXT_SOURCE = "both"        # report + search 결합
TEXT_POOL = "avg"
PROMPT_WEIGHT = 0.1


def _seed_runs(mname, sp, H, domain, device, records):
    """3-seed 학습/평가. seed별 4지표를 records에 append하고 seed 평균 dict 반환."""
    nvar = len(sp["var_cols"])
    tidx = sp["target_idx"]
    per_seed = []
    for seed in SEEDS:
        cfg = ModelConfig(seq_len=L, pred_len=H, enc_in=nvar, c_out=nvar,
                          use_text=True, llm_model=LLM, d_llm=D_LLM, prompt_weight=PROMPT_WEIGHT)
        model = build_model(mname, cfg)
        model = train_model(model, sp["train"], sp["val"], epochs=10, lr=1e-4,
                            patience=5, lradj="type1", device=device, seed=seed)
        p, t, m, s = predict(model, sp["test"], device=device)
        mt = compute_metrics(p, t, target_idx=tidx, mean=m, std=s)
        records.append({"model": mname, "domain": domain, "horizon": H,
                        "seed": seed, **mt})
        per_seed.append(mt)
    return {k: mean(r[k] for r in per_seed) for k in METRICS}


def _save_csv(records):
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"indomain_{stamp}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "domain", "horizon", "seed"] + METRICS)
        w.writeheader()
        w.writerows(records)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="In-domain baseline runner (unimodal + multimodal)")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="실행할 모델 목록 (registry 등록명)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    models, device = args.models, args.device
    need_text = any(m.startswith("MM-TSFlib") for m in models)

    print("=== In-domain Baseline (통합: unimodal + 멀티모달, 3-seed 평균) ===")
    print(f"models = {models}")
    print(f"device={device}, text={LLM+'('+TEXT_SOURCE+'/'+TEXT_POOL+')' if need_text else 'off'}"
          f", prompt_weight={PROMPT_WEIGHT}")
    print(f"L={L}, horizons={HORIZONS}, normalize=dataset, lr=1e-4/type1/10ep\n")

    # (H, domain) -> splits. 다변량 입력 + (멀티모달 있으면) 텍스트 임베딩 부착.
    splits = {(H, d): build_in_domain(d, L, H, multivariate=True, text=need_text,
                                      llm=LLM, text_source=TEXT_SOURCE,
                                      text_pool=TEXT_POOL, device=device)
              for H in HORIZONS for d in MONTHLY_DOMAINS}

    nvar0 = {d: len(splits[(HORIZONS[0], d)]["var_cols"]) for d in MONTHLY_DOMAINS}
    print("입력 변수 수: " + ", ".join(f"{d}={nvar0[d]}" for d in MONTHLY_DOMAINS) + "\n")

    records = []
    avg = {}
    for mname in models:
        for d in MONTHLY_DOMAINS:
            for H in HORIZONS:
                avg[(mname, d, H)] = _seed_runs(mname, splits[(H, d)], H, d, device, records)

    for metric in METRICS:
        for mname in models:
            print(f"\n#########  {mname}  (OT {metric.upper()}) #########")
            print(f"{'domain':12s} " + "".join(f"{'H='+str(H):>9s}" for H in HORIZONS))
            for d in MONTHLY_DOMAINS:
                cells = [avg[(mname, d, H)][metric] for H in HORIZONS]
                print(f"{d:12s} " + "".join(f"{c:9.3f}" for c in cells))

    out_path = _save_csv(records)
    print(f"\n[saved] {out_path}")
    print(f"  {len(records)} rows = {len(models)}모델 × {len(MONTHLY_DOMAINS)}도메인 "
          f"× {len(HORIZONS)}horizon × {len(SEEDS)}seed")
    print("\n(MSE/MAE/RMSE=표준화 공간, MAPE=원 스케일. test=각 도메인 마지막 20%, "
          "다변량 입력·OT 평가. 멀티모달 텍스트=look-back 마지막 달 report+search)")


if __name__ == "__main__":
    main()
