# 04. 평가 프로토콜

메커니즘 검색이 실제로 예측을 개선하는지를 측정한다. 평가는 전부 **in-domain**(각 도메인 내 시간순 split)에서 이뤄진다.

## 셋업

- L=8, H={6,8,10,12}, per-domain global 표준화, 3-seed 평균.
- 평가 = 각 도메인 test split(마지막 20%). 타깃 = OT.
- **HE 누수 방지**: test 시점 `t`의 인스턴스는 `max(evidence_docs.year) < year(t)`인 HE entry만 검색 대상. DK는 필터 없음.

## Metric

| Metric | 설명 |
|--------|------|
| MSE / MAE / RMSE | 표준화 공간 |
| MAPE | 원 스케일 역정규화 |

OT 기준. (`src/evaluation/metrics.py`)

## 베이스라인

| 계열 | 모델 | 텍스트 사용 |
|------|------|------|
| Numerical-only | DLinear, PatchTST, iTransformer, Transformer, Autoformer, Informer, FEDformer | 없음 |
| Aligned-text | MM-TSFlib (frozen LLM 임베딩 + 융합) | 정렬 텍스트 |
| **Ours** | Mechanism-augmented (local branch + global mechanism retrieval) | 정렬 텍스트 + 검색된 메커니즘 |

- Numerical-only = local branch의 TS encoder만 (7 백본). 참조선.
- Aligned-text(MM-TSFlib) = 우리 local branch에 대응하는 기존 멀티모달 베이스라인.
- Ours = local + global mechanism branch. 개선분이 **메커니즘 검색의 순수 기여**.

> In-domain sanity check: 각 모델의 in-domain 성능을 원 논문 보고치와 비교(±5%). MM-TSFlib 데이터로 우리 파이프라인이 논문을 재현함을 이미 검증.

## Ablation — 메커니즘 검색의 효과 분해

부품을 하나씩 빼거나 바꿔 각 요소의 기여를 확인한다 (full_method §29).

| Ablation | 무엇을 검증 |
|----------|------------|
| **Without mechanism branch** | 메커니즘 없이 local branch만 → global branch 전체 기여 |
| **Raw retrieved text vs mechanism reasoning** | 검색 원문 그대로 vs reasoning 거친 메커니즘 → 추론 단계 기여 |
| **Event retrieval vs mechanism retrieval** | 이벤트 인스턴스 검색 vs 메커니즘 검색 → 검색 *대상*의 차이 |
| **Cross-attention without gate** | gate 유무 → 게이팅 기여 |
| **Without forecast-aware query adaptation** | query 규칙 고정 vs 적응 → 적응의 기여 |

**핵심 ablation은 event vs mechanism retrieval** — 검색 대상만 바꾸고 나머지를 고정해, 성능 차이가 "메커니즘 검색"에서 온다는 본 논문의 주장("We retrieve mechanisms, not events")을 직접 뒷받침한다.

## Memorization 분석 (robustness)

CiK 방식:
- **Setup-Clean**: 원본 시계열
- **Setup-Noise**: Gaussian noise σ=3% 추가

두 setup의 성능 차이로 모델이 수치를 암기했는지 vs 실제 메커니즘 신호를 쓰는지 정량화. limitation으로 솔직히 보고.

## "X 모델을 평가했다" 기준

| 변경 범위 | 표기 |
|-----------|------|
| Dataloader만 수정 | "X 평가" |
| 입력 텍스트 형식 변경 | 원본 + 변종 둘 다 보고 |
| 모델 내부 로직 수정 | "X-variant"로 명명 |
