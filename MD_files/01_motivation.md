# 01. 연구 동기 및 차별점

## 핵심 문제의식

### 질문의 재정의

기존 연구는 multimodal 모델이 in-domain에서 unimodal을 이긴다는 사실을 보였다. 우리가 답하고 싶은 질문은 그 한 단계 앞에 있다. **모델이 이벤트를 이해한다는 게 무엇인가, 그리고 현재 방법들은 어디까지 와 있는가.**

### 이벤트 이해를 어떻게 정의할 것인가

세 조건으로 정의한다.

**(a) Locality.** 이벤트가 시계열에 영향을 주는 구간(ROI)을 식별할 수 있어야 한다. 전 구간 평균 MSE만 줄이는 모델은 "어디서" 이벤트가 작동하는지 모른 채 평균만 맞춘 것이다.

**(b) Generalization.** 학습 시 본 (텍스트, 시계열) 쌍의 표면 패턴이 아니라, 이벤트가 어느 변수에 어떤 방식으로 작용하는지에 대한 표현을 가져야 한다. 그래야 학습 때 본 적 없는 도메인의 이벤트에도 텍스트가 도움이 된다.

**(c) Reasoning over retrieval.** 시계열 수치에 작은 perturbation을 가해도 텍스트로 얻은 우위가 살아남아야 한다. 학습 corpus의 텍스트-시계열 co-occurrence를 그대로 끌어 쓰는 retrieval shortcut은 perturbation을 못 견딘다.

셋 모두를 만족해야 "이벤트 이해"라고 부를 수 있다.

### 현재 방법들은 무엇을 학습하는가

**Case-based retrieval (VoT의 HIC, FNF의 reflection).** 학습 데이터에서 (텍스트, 시계열 trajectory) 쌍의 KB를 만들어 두고, 추론 시 유사 케이스를 retrieve해 사용한다. 작동 가설은 "비슷한 텍스트가 비슷한 미래로 이어진다". 그런데 (텍스트, 시계열) 매핑 자체가 도메인 의존적이라는 한계가 있다. 같은 "금리 인상" 텍스트라도 무역수지와 농산물 가격에서 매개 메커니즘이 다른데, retrieval은 이걸 구분하지 못한다.

**LLM-conditioned (Time-LLM 등).** 사전학습 LLM 표현으로 시계열 모델을 조건화한다. LLM이 가진 자연어 이해를 끌어 쓰지만, **LLM 표현을 시계열로 보내는 mapping**은 어차피 학습 corpus에서 end-to-end로 학습된다. 추상적 언어 지식이 자동으로 "허리케인이 실업률에 어떻게 작용하는지"로 번역되지 않는다.

**Architectural fusion (MM-TSFlib, DualTime).** 텍스트와 시계열을 cross-attention 등으로 합친다. fusion 패턴이 학습 분포 위에서 학습되는 한, 위 두 접근과 같은 한계를 공유한다.

요약하면, 현재 방법들은 이벤트의 의미를 도메인-agnostic하게 표현하는 메커니즘을 갖고 있지 않다. 학습 corpus의 텍스트-시계열 co-occurrence 위에서 작동하는 mapping을 학습한다. 이를 "암기"로 단순화하면 부정확하다. 더 정확한 이름은 **in-distribution shortcut**이다. 학습 분포 안에서는 잘 작동하지만, 분포를 벗어나면 보장이 없는 학습이다.

### 왜 in-domain 평가로는 안 잡히는가

in-distribution shortcut과 "진짜 이해"는 in-domain에서 관측상 동일한 답을 낸다. 두 가설은 학습 분포 밖, 즉 새로운 도메인이나 새로운 이벤트가 등장할 때만 갈라진다. 현존 평가 체계는 이 분리를 강제하지 않으므로 둘을 구분할 수 없다. LLM 연구의 "understanding vs pattern matching" 논쟁과 같은 구조다.

### 진단의 세 축과 우리의 결과물

위 세 조건 (a)(b)(c) 각각을 측정할 수 있도록 EATF를 구성했다. 진단과 결과물은 다음과 같이 1:1 대응한다.

**(a) Locality → ROI 라벨 + ROI/non-ROI MSE 분리.**
도메인별로 텍스트가 가리키는 이벤트의 영향 구간(ROI)을 라벨링해서 제공한다. 평가 시 ROI MSE와 non-ROI MSE를 분리하면, 모델이 텍스트를 "이벤트 발생 구간에서" 활용하는지 측정할 수 있다. 라벨링은 LLM duration extraction → CPD validation (IoU ≥ 0.5) → human adjudication의 hierarchical pipeline으로 수행한다.

**(b) Generalization → text-knowledge-transfer protocol.**
단순 LODO(서로 다른 도메인의 시계열을 source/target으로 swap)는 시계열 동역학 자체가 다른 도메인끼리 trivial하게 망가지므로 유효한 진단이 못 된다. 우리는 **target 도메인의 시계열은 고정한 채, 모델이 학습한 텍스트 지식의 출처(source 도메인)만 변화**시키는 1→4 protocol을 설계했다. 이러면 텍스트 모달리티가 가져온 우위만 cross-domain에서 따로 측정된다.

**(c) Reasoning over retrieval → memorization-controlled splits.**
CiK 방식대로 clean / additive-noise variants에서 동일 평가를 반복한다. 텍스트로 인한 우위가 perturbation을 못 견디고 사라진다면, 그 우위의 상당 부분은 retrieval shortcut의 작동이다.

세 진단 모두에서 우위를 유지하는 모델만이 이벤트 이해를 학습했다고 주장할 수 있다. 어느 하나라도 실패하면 그 모델의 이벤트 처리는 in-distribution shortcut에 가깝다.

### 현존 벤치마크는 왜 부족한가

- **Time-MMD** (NeurIPS 2024): 9개 도메인 자연 텍스트 + 시계열 alignment를 제공하지만 ROI 라벨도 cross-domain protocol도 없다. (a), (b) 측정 불가.
- **CiK** (ICML 2025): ROI metric (RCRPS)을 도입했지만 텍스트가 synthetic이다. 자연어 표현의 다양성, 모호성, 암묵성이 제거되어 텍스트 모달리티의 본질적 어려움이 우회된다.
- **VoT, Time-LLM 등 모델 측 평가**는 in-domain 단일 도메인에 국한된다. (b)는 구조적으로 측정 불가.

EATF는 세 진단을 자연 텍스트 위에서 동시에 가능하게 만든다. Cross-domain은 (b)를 위한 도구이며, 그 자체가 목적이 아니다.

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

데이터셋 논문이 입증하려는 것은 (a)(b)(c) 진단 위에서 표현된다.

1. **(a) Locality는 in-domain에서 통과한다.** 현재 multimodal 모델들은 in-domain ROI에서 텍스트를 활용하고 있는 것으로 보일 가능성이 높다(ROI MSE < non-ROI MSE 차이 형태로). 즉 Locality 자체는 약하게나마 만족한다.

2. **(b) Generalization은 무너진다 (핵심 발견).** text-knowledge-transfer protocol 하에서 모델이 다른 source 도메인의 텍스트 지식을 받았을 때, multimodal 우위가 상당 부분 소실될 것으로 예상한다. 특히 case-based 계열(VoT의 HIC, FNF의 reflection)이 가장 크게 무너질 것으로 가정한다. 이게 in-distribution shortcut 가설의 직접 증거.

3. **(c) Reasoning over retrieval도 부분적으로 실패한다.** clean/noise variants에서 multimodal 우위의 일정 부분이 perturbation을 못 견디고 사라질 것으로 본다. 비율은 architecture에 따라 다를 것이고, retrieval 기반일수록 크게 깎인다는 가설.

4. **Architecture family별 실패 패턴이 다르다.** case-based는 (b)에서, retrieval-heavy는 (c)에서, LLM-conditioned/fusion은 (b)에서 각각 다르게 무너질 것. 이 분포 자체가 architectural 의사결정에 시사점을 준다.

이 발견들은 메인 연구(메커니즘 기반 접근)의 motivation으로 직결된다. 셋 모두에 robust한 모델을 만들려면 in-distribution shortcut 너머로 가야 하며, 그게 메커니즘 기반 접근이 풀고자 하는 문제다.
