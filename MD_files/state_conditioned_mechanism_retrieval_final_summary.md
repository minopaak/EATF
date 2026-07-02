# State-Conditioned Mechanism Retrieval for Multimodal Time-Series Forecasting

## 1. 핵심 문제의식

기존 시계열 예측은 주어진 과거 수치 관측값으로부터 미래 값을 외삽하는 문제로 정의되는 경우가 많다.

\[
\mathbf{x}_{t-L+1:t} \rightarrow \mathbf{y}_{t+1:t+H}
\]

하지만 실제 세계의 시계열은 과거 수치 패턴만으로 결정되지 않는다. 수요, 가격, 전력 사용량, 교통량, 질병 발생률, 공급망 지표 등은 외부 사건과 도메인 메커니즘에 의해 크게 변화할 수 있다.

예를 들어 다음과 같은 사건들은 미래 trajectory를 바꾼다.

- 팬데믹
- 항만 폐쇄
- 노동 파업
- 정책 변화
- 공급망 지연
- 기상이변
- 전쟁 또는 지정학적 충격
- 시장 수요 변화
- 원자재 가격 변동

따라서 현실의 forecasting에서는 단순히 과거 numerical window를 보는 것만으로는 충분하지 않다. 중요한 정보는 현재 관측창 안에 직접 드러나지 않거나, 텍스트에 표면적으로만 나타나거나, 과거의 다른 사건과 유사한 mechanism을 통해 간접적으로 작동할 수 있다.

이 논문의 출발점은 다음과 같다.

> Accurate forecasting often depends on information that is not contained in the observed numerical window.

즉, 예측에 필요한 것은 단순한 temporal pattern extrapolation이 아니라, 현재 상태가 미래 trajectory로 전이되는 방식을 설명할 수 있는 external mechanism knowledge이다.

---

## 2. 기존 멀티모달 시계열 예측 연구의 흐름

최근 연구들은 numerical-only forecasting의 한계를 보완하기 위해 text, event, news, domain description, endogenous signal 등을 함께 활용한다. 대표적으로 TaTS, FNTF, VoT와 같은 연구들이 있다.

이 연구들은 크게 세 방향으로 볼 수 있다.

### 2.1 Aligned Text 기반 접근

TaTS류 연구는 시계열과 함께 제공되는 텍스트를 auxiliary input으로 사용한다.

일반적인 형태는 다음과 같다.

\[
(\mathbf{x}_{t-L+1:t}, \mathcal{T}_{t-L+1:t}) \rightarrow \mathbf{y}_{t+1:t+H}
\]

여기서 \(\mathcal{T}\)는 timestamp, segment, patch, instance 등에 정렬된 text이다.

이 접근은 다음을 잘 수행한다.

- numerical sequence와 co-occurring text의 결합
- timestamp-level 또는 patch-level semantic signal 반영
- text가 직접적으로 예측 대상과 연결되어 있을 때의 성능 향상

하지만 한계도 있다.

- text가 반드시 forecasting mechanism을 설명하지는 않는다.
- timestamp에 붙은 text가 미래 변화의 원인이나 작동 경로를 담고 있지 않을 수 있다.
- 표면적으로 유사한 text가 서로 다른 미래 변화를 만들 수 있다.
- 표면적으로 다른 text가 동일한 mechanism을 통해 유사한 미래 변화를 만들 수 있다.

즉, aligned text는 중요한 정보원이지만, 그 자체가 mechanism-level reasoning을 보장하지는 않는다.

### 2.2 Event-driven LLM Forecasting

FNTF와 VoT류 연구는 news나 event를 활용하여 미래 시계열에 미치는 영향을 LLM이 추론하도록 한다.

일반적인 흐름은 다음과 같다.

\[
\text{news/event} \rightarrow \text{event reasoning} \rightarrow \text{forecast}
\]

이 접근은 다음을 잘 수행한다.

- 외부 뉴스나 이벤트를 forecasting 과정에 반영
- LLM을 활용한 event impact reasoning
- historical event, current event, future trajectory 사이의 연결 추론
- reflection을 통한 reasoning 또는 filtering 개선

하지만 이 접근에서도 중요한 한계가 남는다.

- retrieval target이 주로 news, event, historical case, text instance에 머문다.
- 현재 forecasting state에 필요한 mechanism 자체를 명시적으로 찾지는 않는다.
- event similarity와 mechanism similarity를 구분하지 못할 수 있다.
- retrieval된 event가 현재 numerical state와 어떻게 결합되어야 하는지 불명확할 수 있다.

즉, event를 가져오는 것과 mechanism을 가져오는 것은 다르다.

---

## 3. 본 논문의 핵심 문제: State--Mechanism Gap

이 논문은 기존 multimodal forecasting의 한계를 **state--mechanism gap**으로 정의한다.

현재 forecasting instance는 단순한 numerical window가 아니라, 다음과 같은 numerical-textual state로 볼 수 있다.

\[
\mathcal{S}_t = (\mathbf{x}_{t-L+1:t}, \mathcal{E}_{t-L+1:t}, \mathcal{I}_{t-L+1:t}, \tau, H)
\]

여기서:

- \(\mathbf{x}_{t-L+1:t}\): 과거 numerical observations
- \(\mathcal{E}_{t-L+1:t}\): exogenous text  
  예: 뉴스, 정책 발표, 시장 이벤트, 기상 정보, 외부 충격
- \(\mathcal{I}_{t-L+1:t}\): endogenous text  
  예: 내부 로그, 시스템 설명, 도메인 메모, 운영 기록
- \(\tau\): target specification  
  예: 예측 대상 변수, 도메인, 단위, 지역, 품목
- \(H\): forecast horizon

이 state는 현재 상황을 설명하지만, 반드시 미래 변화를 설명하는 mechanism을 포함하지는 않는다.

즉, 현재 state와 미래 trajectory 사이에는 다음과 같은 빈틈이 존재한다.

> The observed forecasting state does not explicitly specify the mechanism by which the current condition evolves into the future target trajectory.

이것이 state--mechanism gap이다.

한국어로 정리하면 다음과 같다.

> 현재 수치-텍스트 상태는 주어져 있지만, 이 상태가 어떤 작동 원리를 통해 미래 변화로 이어지는지는 명시적으로 주어져 있지 않다.

---

## 4. Surface Event와 Mechanism의 차이

이 논문의 가장 중요한 관찰은 다음이다.

> Forecasting requires mechanism similarity, not merely textual or event similarity.

표면적으로 유사한 사건이라고 해서 같은 미래 변화를 만드는 것은 아니다. 반대로 표면적으로 다른 사건이라도 동일한 mechanism을 통해 유사한 미래 trajectory를 만들 수 있다.

### 4.1 표면은 다르지만 mechanism이 같은 경우

다음 사건들은 겉으로는 서로 다르다.

- 팬데믹
- 항만 폐쇄
- 노동 파업
- 전쟁으로 인한 물류 차질
- 대규모 자연재해

하지만 forecasting 관점에서는 유사한 mechanism을 만들 수 있다.

\[
\text{supply disruption} \rightarrow \text{inventory shortage} \rightarrow \text{delayed demand response} \rightarrow \text{price increase}
\]

즉, 텍스트 표면은 다르지만 미래 trajectory에 작동하는 구조는 비슷할 수 있다.

### 4.2 표면은 비슷하지만 mechanism이 다른 경우

반대로 모두 “demand recovery”라는 표현을 포함하더라도, 현재 state에 따라 미래 변화는 달라질 수 있다.

예를 들어:

- 재고가 부족한 상태의 demand recovery
- 재고가 충분한 상태의 demand recovery
- 가격이 높은 상태의 demand recovery
- 계절적 peak 직전의 demand recovery
- 공급망이 불안정한 상태의 demand recovery

이들은 같은 event label을 가질 수 있지만, forecast trajectory는 달라질 수 있다.

따라서 forecasting에서 중요한 것은 단순히 event text를 찾는 것이 아니라, 현재 forecasting state에 적합한 mechanism을 찾는 것이다.

---

## 5. 본 논문의 문제정의

기존 문제는 다음과 같이 정의된다.

### 5.1 Numerical-only Forecasting

\[
\mathbf{x}_{t-L+1:t} \rightarrow \hat{\mathbf{y}}_{t+1:t+H}
\]

과거 수치 관측값만으로 미래 값을 예측한다.

### 5.2 Text-assisted Forecasting

\[
(\mathbf{x}_{t-L+1:t}, \mathcal{T}_{t-L+1:t}) \rightarrow \hat{\mathbf{y}}_{t+1:t+H}
\]

과거 수치 관측값과 정렬된 텍스트를 함께 사용한다.

### 5.3 Event-driven Forecasting

\[
(\mathbf{x}_{t-L+1:t}, \text{news/event}) \rightarrow \text{reasoning} \rightarrow \hat{\mathbf{y}}_{t+1:t+H}
\]

뉴스나 이벤트를 가져와 LLM이 미래 영향을 추론한다.

### 5.4 본 논문의 문제정의

본 논문은 다음 문제를 다룬다.

\[
\mathcal{S}_t \rightarrow q_t \rightarrow \mathcal{R}_t \rightarrow \hat{\mathbf{y}}_{t+1:t+H}
\]

즉:

\[
\text{forecasting state} \rightarrow \text{mechanism query} \rightarrow \text{retrieved mechanisms} \rightarrow \text{forecast}
\]

문제는 다음과 같이 정의할 수 있다.

> Given a numerical--textual forecasting state, identify and retrieve external mechanism knowledge that explains how the current state may evolve into the future target trajectory.

한국어로는 다음과 같다.

> 현재 수치-텍스트 forecasting state가 주어졌을 때, 이 상태가 미래 target trajectory로 전이되는 과정을 설명할 수 있는 외부 mechanism knowledge를 찾아와 예측에 활용하는 문제이다.

---

## 6. Mechanism Pool

본 논문은 외부 문서나 이벤트를 raw text 그대로 retrieval하지 않는다. 대신 forecasting에 활용 가능한 mechanism summary 형태로 정리된 external mechanism pool을 사용한다.

\[
\mathcal{M} = \{(m_i, a_i, \mathbf{z}_i)\}_{i=1}^{N}
\]

각 entry는 다음으로 구성된다.

- \(m_i\): mechanism summary
- \(a_i\): source evidence 또는 provenance
- \(\mathbf{z}_i\): retrieval embedding

여기서 \(m_i\)는 단순 event summary가 아니다.

나쁜 예시는 다음과 같다.

> 항만 파업이 발생했다.

이것은 단순 event description이다.

좋은 mechanism summary는 다음과 같다.

> 항만 파업은 물류 지연을 유발하고, 재고가 낮은 품목에서는 단기적인 공급 부족과 주문 지연을 만든다. 이후 대체 공급원이 확보되거나 재고가 회복되면서 수요와 가격은 지연된 회복 패턴을 보일 수 있다.

이 문장은 단순히 사건을 요약하는 것이 아니라, 사건이 미래 시계열에 영향을 미치는 작동 경로를 설명한다.

즉, mechanism pool의 목적은 다음이다.

> Store reusable forecasting mechanisms rather than isolated event descriptions.

---

## 7. State-to-Mechanism Query Writer

Mechanism pool이 있더라도, 현재 forecasting state에서 어떤 mechanism을 찾아야 하는지 query를 잘 만들어야 한다.

단순 event query는 다음처럼 표면 키워드 중심이 된다.

> port strike shipping delay demand

하지만 mechanism query는 다음처럼 state, target, horizon, effect pathway를 포함해야 한다.

> mechanisms by which logistics disruptions under low inventory conditions affect short-term retail demand over a two-week horizon

따라서 본 논문은 현재 forecasting state를 mechanism-seeking query로 변환하는 query writer를 둔다.

\[
q_t = Q_{\rho}(\mathcal{S}_t)
\]

여기서:

- \(\mathcal{S}_t\): 현재 forecasting state
- \(Q_{\rho}\): state-to-mechanism query writer
- \(q_t\): generated mechanism query
- \(\rho\): query-writing rule 또는 prompt policy

이 query writer의 역할은 단순히 관련 단어를 뽑는 것이 아니다. 현재 state에서 미래 변화를 설명할 수 있는 mechanism을 찾도록 retrieval intent를 재구성하는 것이다.

즉, 핵심은 다음이다.

> The key difference is not the use of retrieval itself, but what the query is designed to retrieve.

---

## 8. Mechanism Retrieval

생성된 query \(q_t\)를 사용하여 mechanism pool에서 관련 mechanism을 검색한다.

\[
\mathcal{R}_t = \operatorname{Retrieve}(q_t, \mathcal{M}, K)
\]

여기서:

- \(\mathcal{R}_t\): retrieved mechanisms
- \(\mathcal{M}\): mechanism pool
- \(K\): 검색할 mechanism 수

이때 retrieval 대상은 다음이 아니다.

- raw news
- raw document
- timestamp-level text
- historical event instance
- similar example only

검색 대상은 다음이다.

- 현재 state와 미래 trajectory 사이의 transition을 설명하는 mechanism summary

따라서 본 논문의 retrieval은 다음 문장으로 요약할 수 있다.

> We retrieve mechanisms, not events.

---

## 9. Mechanism-aware Forecasting

검색된 mechanism \(\mathcal{R}_t\)는 forecasting model의 추가 context로 사용된다.

\[
\hat{\mathbf{y}}_{t+1:t+H}
= G_{\theta}(\mathbf{x}_{t-L+1:t}, \mathcal{E}_{t-L+1:t}, \mathcal{I}_{t-L+1:t}, \mathcal{R}_t, \tau, H)
\]

여기서 \(G_{\theta}\)는 forecasting model이다.

중요한 점은 retrieved mechanism이 timestamp-level annotation이 아니라는 것이다. 즉, 특정 시점 하나에 붙는 local text가 아니라, 전체 forecasting state를 해석하기 위한 global reasoning context이다.

기존 aligned text는 보통 다음과 같은 방식이다.

\[
x_t \leftrightarrow text_t
\]

반면 본 논문에서는 다음과 같다.

\[
\mathcal{S}_t \leftrightarrow \mathcal{R}_t
\]

즉, retrieved mechanism은 현재 forecasting state 전체에 대한 해석적 context로 작동한다.

---

## 10. Forecast-aware Query Adaptation

Mechanism retrieval은 query에 민감하다. 처음 생성된 query가 표면 이벤트 중심이면, retrieval도 surface similarity에 끌릴 수 있다.

예를 들어 다음 query는 너무 표면적이다.

> pandemic demand drop

이 query는 팬데믹 관련 문서는 잘 찾을 수 있지만, 실제 forecasting에 필요한 mechanism을 충분히 찾지 못할 수 있다.

더 좋은 query는 다음과 같다.

> mobility restriction causing offline demand collapse, online substitution, and delayed normalization under changing consumer behavior

따라서 본 논문은 validation feedback을 이용해 query-writing rule을 개선한다.

전체 흐름은 다음과 같다.

\[
\text{state} \rightarrow \text{query} \rightarrow \text{retrieval} \rightarrow \text{forecast} \rightarrow \text{error}
\]

Validation set에서 high-error cases를 분석하고, query-writing rule을 수정한다.

\[
\rho^{(r+1)} = \operatorname{Reflect}(\rho^{(r)}, \mathcal{D}_{val}, \mathcal{E}_{error})
\]

여기서:

- \(\rho^{(r)}\): 현재 round의 query-writing rule
- \(\mathcal{D}_{val}\): validation set
- \(\mathcal{E}_{error}\): high-error forecasting cases
- \(\rho^{(r+1)}\): 개선된 query-writing rule

이 과정의 핵심은 forecasting error를 통해 query rule을 개선한다는 점이다.

FNTF의 reflection이 event filtering이나 reasoning trace 개선에 가깝다면, 본 논문의 reflection은 다음을 개선한다.

> How to convert a forecasting state into a mechanism-seeking query.

즉, reflection의 대상이 다르다.

---

## 11. 전체 Pipeline

전체 방법론은 다음 순서로 정리된다.

1. 현재 forecasting instance에서 numerical-textual state \(\mathcal{S}_t\)를 구성한다.
2. State-to-mechanism query writer \(Q_{\rho}\)가 \(\mathcal{S}_t\)를 mechanism query \(q_t\)로 변환한다.
3. Query \(q_t\)를 사용해 mechanism pool \(\mathcal{M}\)에서 관련 mechanism \(\mathcal{R}_t\)를 검색한다.
4. Forecasting model \(G_{\theta}\)는 numerical history, textual inputs, retrieved mechanisms를 함께 사용해 future trajectory를 예측한다.
5. Validation error를 분석하여 query-writing rule \(\rho\)를 반복적으로 개선한다.

전체 구조는 다음과 같다.

\[
\mathcal{S}_t
\xrightarrow{Q_{\rho}}
q_t
\xrightarrow{\operatorname{Retrieve}}
\mathcal{R}_t
\xrightarrow{G_{\theta}}
\hat{\mathbf{y}}_{t+1:t+H}
\]

---

## 12. 기존 연구 대비 차별점

| 구분 | 기존 접근 | 한계 | 본 논문 |
|---|---|---|---|
| Numerical-only forecasting | 과거 수치 window 기반 예측 | 외부 mechanism 반영 어려움 | numerical-textual state를 사용 |
| Aligned text forecasting | timestamp/patch/instance text 결합 | text가 mechanism을 설명하지 않을 수 있음 | state-level mechanism retrieval |
| Event-driven forecasting | news/event를 가져와 LLM reasoning | event similarity와 mechanism similarity를 구분하기 어려움 | mechanism summary를 retrieval target으로 설정 |
| Historical case retrieval | 유사 과거 사례 검색 | surface similarity에 의존 가능 | 현재 state에 맞는 mechanism similarity 중심 검색 |
| Reflection-based forecasting | reasoning/filtering 개선 | query가 무엇을 찾도록 설계되는지 명시 부족 | forecast-aware query-writing rule adaptation |

핵심 차별점은 다음 한 문장으로 정리된다.

> Multimodal time-series forecasting requires a retrieval target beyond aligned text and event instances: mechanism-level knowledge conditioned on the current forecasting state.

---

## 13. Contribution 정리

본 논문의 기여는 다음 세 가지로 정리할 수 있다.

### Contribution 1. State--Mechanism Gap 정의

기존 multimodal time-series forecasting은 numerical history와 aligned text 또는 event input을 결합하는 데 집중해왔다. 본 논문은 이러한 접근이 현재 forecasting state와 미래 trajectory를 연결하는 mechanism-level knowledge를 명시적으로 다루지 못한다는 점을 지적하고, 이를 state--mechanism gap으로 정의한다.

핵심 문장:

> We identify the state--mechanism gap in multimodal time-series forecasting: the mismatch between the observed numerical--textual forecasting state and the external mechanisms needed to explain its future evolution.

### Contribution 2. State-Conditioned Mechanism Retrieval 제안

본 논문은 raw event나 aligned text가 아니라, forecasting state에 조건화된 mechanism summary를 retrieval target으로 설정한다. 이를 위해 external mechanism pool을 구성하고, 현재 state를 mechanism-seeking query로 변환하는 State-to-Mechanism Query Writer를 제안한다.

핵심 문장:

> We formulate mechanism retrieval as a state-conditioned query-writing problem, where the retrieval target is not an event or document, but a reusable mechanism that explains how the current state may evolve.

### Contribution 3. Forecast-aware Query Adaptation 제안

Mechanism retrieval은 query에 민감하므로, 본 논문은 validation forecasting error를 활용하여 query-writing rule을 반복적으로 개선한다. 이를 통해 단순 surface event retrieval이 아니라, forecasting 성능에 실제로 도움이 되는 mechanism retrieval로 query를 조정한다.

핵심 문장:

> We introduce forecast-aware query adaptation, which refines the query-writing rule using high-error validation cases so that retrieval is optimized toward forecasting-relevant mechanisms rather than surface-level textual similarity.

---

## 14. 논문 Introduction 전개 방향

Introduction은 다음 순서로 전개하는 것이 좋다.

### Paragraph 1. 문제의식 제시

첫 문장은 forecasting의 교과서적 정의가 아니라, 바로 한계를 찌르는 문장으로 시작한다.

추천 시작 문장:

> Accurate forecasting often depends on information that is not contained in the observed numerical window.

이후 numerical-only forecasting의 한계와 외부 mechanism의 필요성을 설명한다.

### Paragraph 2. 기존 multimodal forecasting 연구 정리

TaTS, FNTF, VoT류 연구를 압축적으로 정리한다.

- TaTS: aligned text를 auxiliary input으로 활용
- FNTF: news/event filtering, event reasoning, reflection
- VoT: event-driven reasoning, historical in-context learning, endogenous/exogenous alignment, multi-level fusion

여기서 중요한 것은 기존 연구를 단순히 나열하지 않는 것이다. 반드시 다음 문제로 연결해야 한다.

> These methods improve the use of textual information, but they still largely retrieve or align textual/event instances rather than the mechanisms that explain future state transitions.

### Paragraph 3. State--Mechanism Gap 제시

여기서 본 논문의 gap을 제시한다.

핵심 문장:

> The missing object is not text itself, but mechanism-level knowledge that explains how a current forecasting state may evolve into a future trajectory.

그리고 surface event와 mechanism의 차이를 설명한다.

### Paragraph 4. 방법 개요

State-conditioned mechanism retrieval을 소개한다.

핵심 구성 요소:

- numerical-textual forecasting state
- external mechanism pool
- state-to-mechanism query writer
- mechanism retrieval
- mechanism-aware forecasting
- forecast-aware query adaptation

### Paragraph 5. Contributions

세 가지 contribution을 명확히 제시한다.

1. State--mechanism gap
2. State-conditioned mechanism retrieval
3. Forecast-aware query adaptation

---

## 15. 논문에서 피해야 할 표현

다음 표현들은 약하거나 일반적이므로 피하는 것이 좋다.

### 약한 표현

- Time-series forecasting is often formulated as...
- In many real-world domains, however,...
- Recent studies have shown...
- Text can provide useful information...
- This is important because...
- We propose a novel framework...

이 표현들은 너무 흔하고, top conference 논문 첫 문단에서 강한 문제의식을 만들기 어렵다.

### 더 나은 표현

- Accurate forecasting often depends on information that is not contained in the observed numerical window.
- The relevant context is not always the co-occurring text, but the mechanism that governs the state transition.
- Surface event similarity can be misleading for forecasting.
- The key retrieval target should be a mechanism, not an event instance.
- We formulate mechanism retrieval as a state-conditioned query-writing problem.
- Forecasting errors provide supervision for refining what the query should retrieve.

---

## 16. 용어 정리

| 용어 | 의미 |
|---|---|
| Forecasting state | 현재 예측 instance를 구성하는 numerical history, textual evidence, target specification, horizon의 결합 |
| External mechanism | 현재 state가 미래 trajectory로 변화하는 작동 경로를 설명하는 외부 지식 |
| Mechanism summary | event 자체가 아니라 event가 future target에 영향을 주는 pathway 요약 |
| Mechanism pool | reusable mechanism summaries를 저장한 retrieval corpus |
| State-to-mechanism query | 현재 forecasting state로부터 필요한 mechanism을 찾기 위해 생성된 query |
| State--mechanism gap | 관측된 forecasting state와 미래 변화를 설명할 mechanism knowledge 사이의 간극 |
| Forecast-aware query adaptation | validation forecasting error를 이용해 query-writing rule을 개선하는 과정 |

---

## 17. 최종 한 줄 요약

이 논문은 다음 한 문장으로 정리할 수 있다.

> This paper reframes multimodal time-series forecasting as a state-conditioned mechanism retrieval problem, where the goal is to retrieve external mechanism knowledge that explains how the current numerical--textual state may evolve into the future target trajectory.

한국어로는 다음과 같다.

> 본 논문은 멀티모달 시계열 예측을 단순한 텍스트 결합 문제가 아니라, 현재 수치-텍스트 상태가 미래 trajectory로 전이되는 작동 원리를 설명하는 외부 mechanism knowledge를 검색하는 문제로 재정의한다.
