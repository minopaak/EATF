# EATF: Event-Aware Time Series Forecasting via Mechanism Retrieval

Time-MMD 기반 이벤트-인지 시계열 예측 프로젝트. 예측 맥락에 맞는 외부 **메커니즘 지식**을 사전 구축 pool에서 검색해 예측을 조건화한다. 문제 정의는 [MD_files/01_motivation.md](MD_files/01_motivation.md), 전체 방법은 [state_conditioned_mechanism_retrieval_full_method.md](MD_files/state_conditioned_mechanism_retrieval_full_method.md) 참조.

> **We retrieve mechanisms, not events.**

## Setup

```bash
git clone https://github.com/minopaak/EATF.git
cd EATF
uv venv --python 3.11
uv pip install -e .

# (선택) Time-MMD 원본으로 데이터 재구축
mkdir -p clones && cd clones && git clone https://github.com/AdityaLab/Time-MMD.git && cd ..
python build_dataset.py --all
```

`data/processed/*.csv`(5개 도메인 가공본)는 repo에 포함 — 재구축 없이 베이스라인 실행 가능.

```bash
CUDA_VISIBLE_DEVICES=4 python -m src.run       # in-domain 베이스라인 (7 백본 × {uni, MM})
python -m src.report seedavg                    # seed 평균 CSV
python -m src.report latex --metric mse         # LaTeX 표
```

## 프로젝트 구조

```
EATF/
├── build_dataset.py            # 예측 데이터 구축 (Time-MMD → processed CSV)
├── MD_files/                   # 설계 문서 (01_motivation ~ 05_roadmap + method 원본)
├── data/
│   ├── processed/              # 5개 도메인 merged CSV
│   └── text_emb/               # frozen LLM 임베딩 캐시 (gitignore)
├── results/                    # 실험 결과 CSV / LaTeX (gitignore)
├── notebooks/01_eda.ipynb
├── src/
│   ├── run.py                  # in-domain 베이스라인 러너
│   ├── report.py               # seed 평균 / 비교 / LaTeX
│   ├── sweep.py                # prompt_weight ablation
│   ├── data/loader.py          # 윈도우 로더 (build_in_domain)
│   ├── evaluation/metrics.py   # MSE/MAE/RMSE/MAPE
│   ├── training/trainer.py     # 학습/예측 루프
│   └── models/
│       ├── config.py, registry.py, text_encoder.py
│       ├── layers/             # 공통 빌딩 블록 (TSLib/MM-TSFlib 이식, 출처 헤더 표기)
│       └── architectures/      # 7 백본 (DLinear/PatchTST/iTransformer/Transformer/
│                               #   Autoformer/Informer/FEDformer) + mm_fusion
└── clones/                     # 외부 repo (gitignore)
```

**Mechanism pool**은 별도 repo `knowledge_pool`이 생성하고(DK/HE JSON), EATF는 이를 소비만 한다. [MD_files/02_dataset_design.md](MD_files/02_dataset_design.md) §B 참조.

## 진행 상황

- [x] 문제 정의 재정립 (메커니즘-증강 예측)
- [x] 예측 데이터 5개 도메인 구축 (텍스트 시대 trim)
- [x] in-domain 베이스라인 (7 백본 × {unimodal, MM-TSFlib}, L=8, 3-seed) — **local branch 참조선**
- [x] 파이프라인 정합성 검증 (MM-TSFlib 논문 재현)
- [ ] mechanism pool 소비 브리지 + retrieval 인프라 ← **다음**
- [ ] mechanism-aware forecasting 모델 (local + global branch + gated fusion)
- [ ] inner/outer loop 학습
- [ ] 평가·ablation·논문 작성

## 핵심 설계 결정

| 항목 | 결정 |
|------|------|
| 기반 데이터 | Time-MMD (monthly 5개 도메인) |
| 도메인 | Agriculture, Economy, Security, SocialGood, Traffic |
| 시간 범위 | 텍스트 시대로 trim (Agri 523/Eco 447/Sec 309/Social 533/Traffic 531행) |
| 텍스트 | Report + Search (Security는 search-only) |
| Look-back / Horizon | L=8, H={6,8,10,12} |
| 정규화 | per-domain global StandardScaler (RevIN 금지) |
| Mechanism pool | DK(일반 원리) + HE(역사적 사건), `knowledge_pool` 산출 JSON 소비 |
| 검색 대상 | 이벤트가 아니라 메커니즘 `content` |
| Memorization | CiK 방식 (Gaussian noise) |
