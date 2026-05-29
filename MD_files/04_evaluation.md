# 04. 평가 프로토콜 및 베이스라인

## 평가 Protocol

### 3가지 셋업

| 셋업 | 설명 | 목적 |
|------|------|------|
| **In-domain** | 도메인 내 시간순 split | 상한선 측정 |
| **Zero-shot LODO** | 1개 도메인의 텍스트/이벤트 지식으로 학습 → 4개 도메인 평가 (target TS는 그대로, 텍스트/이벤트 reasoning이 transfer되나) | Cross-domain 일반화 핵심 |
| **Few-shot LODO** | 1개 학습 + target N개 샘플 (N=10/50/100) | 현실적 셋업 |

### 왜 1→4 방향?
일반적 LODO와 반대 방향 (N-1 train, 1 holdout). 더 어려운 셋업(1 train, N-1 holdout)으로 case-based 방법의 한계를 더 강하게 입증.

### Cross-domain의 정확한 의미 (중요)
**우리 cross-domain은 단순 TS swap이 아니라 텍스트/이벤트 지식 transfer 테스트다.**

- 평가하는 시계열은 항상 **target 도메인의 TS** (그대로). 모델한테 forecast하라고 주는 입력·출력은 target의 OT.
- 바뀌는 건 **모델이 가진 텍스트/이벤트 지식의 출처** — source 도메인에서 학습된(또는 source의 텍스트/KB를 쓰는) 모델이 target TS를 예측할 때, source-derived 이벤트 reasoning이 도움 되나?
- 핵심 질문: 모델이 **도메인-agnostic한 이벤트 의미**를 배웠나, 아니면 source-text↔source-TS pairing만 외웠나? VoT 같은 case-based 방법은 후자라는 게 우리 가설.

**왜 단순 TS swap은 안 쓰나:** 도메인마다 시계열 동역학이 근본적으로 다름(Agri 가격 vs Economy 무역수지 vs Security 재난보조금). source TS로 학습한 모델이 target TS 못 맞추는 건 trivial(사과를 오렌지에 적용)이라 흥미로운 인사이트가 안 나옴. 우리 contribution은 이벤트(텍스트) 모달리티의 transferability이지 TS 자체의 transferability가 아니므로, 측정도 그에 맞춰야 함.

**TS-only 베이스라인(DLinear/PatchTST)은 cross-domain 헤드라인 비교에 참여하지 않음.** 텍스트 입력이 없어 측정할 게 없기 때문. In-domain 측정에만 의미가 있고, cross-domain 자리에는 TS-swap LODO 결과를 reference로 표시할 수는 있으나 헤드라인은 아님.

## 평가 Track

### Track A: Controlled (사전 라벨 제공)
- 모든 모델이 우리 사전 라벨을 동일하게 받음
- Train + test 모두 적용
- 시계열-텍스트 융합 능력 자체를 fair하게 비교

### Track B: Native (원본 텍스트)
- 모델이 Time-MMD raw fact 텍스트를 자체 파이프라인으로 처리
- 모델의 종합 능력 평가

## Metric

| Metric | 설명 |
|--------|------|
| Overall MSE/MAE | 전체 시점 평균 |
| ROI MSE/MAE | ROI mask 안의 시점만 |
| Non-ROI MSE/MAE | mask 밖 |
| RCRPS | CiK 방식 채택 |
| Cross-domain degradation ratio | CD MSE / In-D MSE (둘 다 target test split·동일 정규화 → std 약분되어 정규화 윈도우에 영향 받지 않음) |

### 핵심 분석 지표
**ROI degradation / Overall degradation** 비율.

1보다 크면 "평상시 패턴은 transfer되지만 이벤트 처리는 transfer 안 됨" 입증. 메인 연구 motivation으로 직결.

## 베이스라인 (6개)

| 모델 | 카테고리 | 코드 변경 범위 |
|------|---------|--------------|
| PatchTST | Unimodal | Dataloader만 |
| DLinear | Unimodal | Dataloader만 |
| MM-TSFlib | Time-MMD native | Dataloader만 |
| VoT | Multimodal SOTA | Dataloader + 텍스트 입력 |
| Time-LLM | LLM-based | Dataloader + 텍스트 입력 |
| DualTime | Multimodal | Dataloader + 텍스트 입력 |

### VoT 변종 평가 (핵심 ablation)
- **VoT-HIC-source**: Source 도메인 KB만 사용
- **VoT-HIC-off**: HIC 끄고 reasoning만
- **VoT-HIC-fewshot**: Target 일부 sample로 KB 일부 구축

이 비교로 "case-based retrieval이 cross-domain에서 도움 안 됨" 정량화.

### FNF 제외
자체 데이터 포맷이라 호환 불가. Related work에서만 언급.

## Memorization 처리

CiK 방식 채택:
- **Setup-Clean**: 원본 시계열
- **Setup-Noise**: Gaussian noise 표준편차 3% 추가

두 setup의 성능 차이로 memorization 영향 정량화.

Fidel-TS 비판 대응:
- 차별점 명시 (event reasoning vs scheduled covariate)
- Memorization 분석을 추가 contribution으로 다룸
- Limitation으로 솔직히 인정

## 베이스라인 모델 사용 시 주의사항

### "X 모델을 평가했다"고 적을 수 있는 기준

| 변경 범위 | 표기 |
|-----------|------|
| Dataloader만 수정 | "X 모델 평가" (떳떳이) |
| 입력 텍스트 형식 변경 | 원본 + 변종 둘 다 보고 |
| 모델 내부 로직 수정 | "X-variant"로 이름 |

### In-domain 성능 검증
모든 모델에 대해 in-domain 성능을 원본 논문 보고치와 비교 (±5% 이내). 이게 sanity check.

## 구현 우선순위

1. **PatchTST 먼저** (가장 단순)
2. **MM-TSFlib** (Time-MMD native라 호환 좋음)
3. **VoT** (핵심 비교 대상)
4. 나머지 (DLinear, Time-LLM, DualTime)
