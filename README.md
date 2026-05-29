# Cross-Domain Event-Aware Time Series Forecasting Benchmark

Time-MMD 기반 cross-domain event-aware forecasting 벤치마크 데이터셋 구축 프로젝트.

## 프로젝트 구조

```
EATF Dataset/
├── README.md                          # 이 파일
├── .gitignore
├── build_dataset.py                   # 데이터 구축 (Phase 1, 모델 학습 이전 단계)
├── MD_files/                          # 설계 문서 (01_motivation ~ 05_roadmap)
├── data/
│   └── processed/                     # 가공된 CSV (5개 도메인)
│       ├── Agriculture_merged.csv
│       ├── Economy_merged.csv
│       ├── Security_merged.csv
│       ├── SocialGood_merged.csv
│       └── Traffic_merged.csv
├── notebooks/
│   └── 01_eda.ipynb                   # EDA
├── reports/
│   └── progress_summary.html          # 진행 요약 슬라이드
├── src/                               # 모델 / 학습 코드 (Phase 2~)
│   ├── data/
│   │   └── loader.py                  #   학습용 로더 (윈도우 / RevIN / LODO)
│   └── models/
│       ├── config.py                  #   ModelConfig (framework)
│       ├── layers.py                  #   공통 빌딩 블록 (framework)
│       ├── registry.py                #   build_model(name, cfg)
│       └── architectures/             #   모델 구현
│           ├── patchtst.py            #     PatchTST
│           └── dlinear.py             #     DLinear
└── clones/                            # 외부 repo (각자 .git, gitignore 처리)
    ├── Time-MMD/                      # 원본 데이터
    ├── Time-Series-Library/          # TSLib (모델 구현 참고용)
    └── MM-TSFlib/                    # Time-MMD native 베이스라인
```

추후 추가 예정: `src/training/`, `src/evaluation/`, `src/annotation/`, `results/`

## 진행 상황

- [x] 연구 방향 및 차별점 설계
- [x] 데이터셋 스코프 결정 (Monthly 5개 도메인 — Climate는 실측 weekly로 제외)
- [x] ROI 정의 및 라벨링 파이프라인 설계
- [x] 평가 프로토콜 설계
- [x] 베이스라인 선정
- [x] 데이터 통합 CSV 생성 (5개 도메인, `data/processed/`, 텍스트 시대 trim 적용)
- [x] TS-only 베이스라인 (DLinear, PatchTST, L=8, 3-seed) 측정
- [x] 파이프라인 정합성 검증 (MM-TSFlib 데이터로 우리 코드 돌려 논문 재현)
- [ ] EDA 노트북 (Phase 1 잔여)
- [ ] **LLM 이벤트 라벨링 파이프라인 구현 (Phase 3) ← 다음**
- [ ] CPD 검증 통합
- [ ] Multimodal 베이스라인 통합 (Phase 5) — cross-domain 텍스트 transfer 헤드라인 실험
- [ ] 분석 및 작성

## 다음 단계

**Phase 3 (ROI 라벨링)부터.** 헤드라인 cross-domain 실험은 단순 TS-swap LODO가 아니라 multimodal 모델의 텍스트/이벤트 지식 transfer 테스트라, multimodal 베이스라인(Phase 5)이 필요하고 그건 ROI 라벨이 전제. 자세한 설계는 [04_evaluation.md](MD_files/04_evaluation.md)·[05_roadmap.md](MD_files/05_roadmap.md) 참조.

## 핵심 설계 결정

| 항목 | 결정 |
|------|------|
| 기반 데이터 | Time-MMD |
| 도메인 | Monthly 5개 (Agriculture, Economy, Security, SocialGood, Traffic) |
| 시간 범위 | **텍스트 시대로 trim** (텍스트 존재 구간; Agri 523/Eco 447/Sec 309/Social 533/Traffic 531행) |
| 텍스트 | Report + Search 둘 다 사용, 컬럼 분리 |
| 텍스트 월 매칭 | Majority-overlap (윈도우가 더 많이 걸친 월; 동률시 빠른 월) |
| Look-back / Horizon | L=8 (Time-MMD monthly), H={6, 8, 10, 12} |
| ROI 추정 | pred 우선 → 없으면 LLM+CPD |
| 평가 트랙 | Track A (사전 라벨) + Track B (원본 텍스트) |
| LODO 방향 | 1개 학습 → 4개 평가 |
| Memorization | CiK 방식 (Gaussian noise) |
