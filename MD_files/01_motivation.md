# 01. 연구 동기 및 차별점

## 핵심 문제의식

시계열 예측에서 외부 이벤트(정책 발표, 자연재해, 시장 충격 등)는 급격한 변동을 유발하는 핵심 요인이다. 최근 multimodal 시계열 연구들(VoT, FNF)이 LLM agent로 텍스트 정보를 활용해 이벤트 처리를 시도했지만 두 가지 한계가 있다.

**한계 1: In-domain 평가만 수행**
- 학습한 도메인에서만 평가하니까 case-based retrieval (VoT의 HIC, FNF의 reflection)이 잘 작동하는 것처럼 보임
- 실제 일반화 능력이 검증된 적 없음

**한계 2: Cross-domain 일반화 능력 미검증**
- 실제 배포 환경에선 학습 분포 밖의 도메인이나 새로운 이벤트가 등장
- 이때 기존 방법들이 어떻게 작동하는지 알 수 없음

이 한계를 해결하려면 **cross-domain 평가가 가능한 event-aware 벤치마크**가 필요하다.

## 연구 구조

두 갈래로 진행:

- **메인 연구**: 메커니즘 기반 event-aware 시계열 예측 (case-based 한계 극복)
- **데이터셋 논문 (본 프로젝트)**: 메인 연구의 motivation을 정량적으로 입증할 평가 기반 제공

데이터셋이 필요한 이유:
1. 기존 방법이 cross-domain에서 무너진다고 주장하려면 그걸 평가할 벤치마크가 필요
2. 현존 벤치마크 중 이를 지원하는 게 없음
3. 데이터셋 자체로도 독립 contribution

## 선행 연구 대비 차별점

### Time-MMD (NeurIPS 2024)
- 한 일: 9개 도메인 자연 텍스트를 시계열과 fine-grained alignment, fact/prediction 분리
- 안 한 일: Event-level annotation, ROI 정의, cross-domain split
- 우리: 이 위에 event 차원 추가

### CiK (ICML 2025)
- 한 일: ROI 기반 metric (RCRPS) 도입
- 한계: 텍스트가 인위적으로 만들어진 것(crafted), cross-domain 평가 없음
- 우리: 자연 텍스트에 cross-domain 평가 추가

### VoT
- 한 일: Time-MMD에서 in-domain SOTA
- 한계: Cross-domain 미검증
- 우리: VoT의 HIC (case-based retrieval)가 cross-domain에서 어떻게 무너지는지 분석이 핵심 포인트

### FNF (NeurIPS 2024)
- 한 일: 영향 분류 (positive/negative + short/long-term)
- 한계: 자체 데이터셋이라 베이스라인 비교 불가
- 우리: Related work에서만 언급, 영향 분류 개념은 ROI 라벨이 차용

### Fidel-TS (2025)
- 한 일: Contamination-free를 위해 scheduled covariate만 사용
- 차이: 우리는 unexpected event reasoning에 집중
- 포지셔닝: 다른 niche, 보완 관계

## 예상 핵심 발견

데이터셋 논문이 입증하려는 4가지:

1. **Multimodal 우위 감소**: In-domain에서 multimodal이 unimodal 대비 X% 우위였는데 cross-domain에선 Y%로 줄어듦 (Y << X)
2. **ROI 특화 성능 저하**: 모든 multimodal 모델에서 ROI degradation > Overall degradation
3. **사전 라벨로도 gap 해소 안 됨**: 라벨 제공해도 cross-domain gap 존재 → 모델이 generic하게 이벤트 활용하는 능력 부족
4. **Case-based 방법의 한계**: VoT의 HIC가 cross-domain에서 도움 안 됨/오히려 해로움

이 발견들이 메인 연구(메커니즘 기반 접근)의 motivation으로 직결.
