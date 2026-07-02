# 01. EATF: 이벤트-인지 시계열 예측 — 문제 정의

## 1. 문제

시계열 예측은 흔히 과거 수치 창을 미래로 외삽하는 문제로 정의된다.

$$
\mathbf{x}_{t-L+1:t} \rightarrow \hat{\mathbf{y}}_{t+1:t+H}
$$

하지만 현실의 시계열 — 무역수지, 유가, 수요, 감염자 수, 교통량 — 은 과거 수치만으로 결정되지 않는다. 팬데믹·항만 폐쇄·정책 변화·공급망 충격 같은 **외부 사건과 도메인 메커니즘**이 미래 궤적을 바꾸며, 이 정보는 관측된 수치 창 안에 들어있지 않다.

> **Accurate forecasting often depends on information that is not contained in the observed numerical window.**

**이벤트-인지 시계열 예측(EATF)** 은 이렇게 수치 창 밖에 있는 사건·메커니즘을 인지해 예측에 반영하는 문제다.

## 2. 기존 접근과 공통 한계

| 접근 | 하는 일 | 한계 |
|---|---|---|
| Numerical-only | 과거 수치 외삽 | 외부 요인 반영 불가 |
| Aligned-text (TaTS류) | 시점에 정렬된 텍스트를 결합 | 정렬 텍스트가 *미래 변화의 작동 원리*를 담지 않음 |
| Event-driven (FNTF·VoT류) | 뉴스·이벤트를 LLM이 추론 | 검색·정렬 대상이 **이벤트/문서 인스턴스**에 머묾 |

이들은 텍스트 활용을 발전시켰지만, 여전히 **텍스트·이벤트 인스턴스를 가져오거나 정렬**할 뿐, 현재 상황이 미래로 전이되는 **작동 원리(메커니즘)** 를 명시적으로 다루지 않는다.

## 3. 핵심 관찰: 이벤트 유사도 ≠ 메커니즘 유사도

표면이 다른 사건도 예측 관점에선 같은 메커니즘을 공유할 수 있고,

```text
팬데믹 / 항만폐쇄 / 파업
  → supply disruption → inventory shortage → delayed demand response → price change
```

표면이 같은 사건도 현재 조건에 따라 다른 궤적을 만든다.

```text
"demand recovery"
  재고 부족 → 품절 위험·지연 회복
  재고 충분 → 즉시 회복
  고가 국면 → 억눌린 회복
```

> **Surface event similarity can be misleading; what forecasting needs is mechanism-level knowledge.**

따라서 이벤트-인지 예측에서 정말 필요한 검색 대상은 *이벤트*가 아니라, 그 사건이 타깃 변수에 작동하는 **메커니즘**이다.

## 4. 문제 정의 — 메커니즘-증강 예측

우리는 EATF를 **외부 메커니즘 지식을 검색해 예측을 조건화하는 문제**로 푼다. 사전 구축된 **mechanism pool**이 주어졌을 때, 현재 예측 맥락(과거 수치 + 정렬 텍스트)에 관련된 메커니즘 서술을 검색하고, 이를 조건으로 미래 궤적을 예측한다.

$$
\big(\mathbf{x}_{t-L+1:t},\ \mathcal{T}\big)\ \longrightarrow\ \underbrace{\mathcal{R}_t}_{\text{pool에서 검색된 메커니즘}}\ \longrightarrow\ \hat{\mathbf{y}}_{t+1:t+H}
$$

검색 대상은 raw 뉴스·문서·시점 텍스트가 아니라, **현재 맥락에서 미래 변화를 설명하는 메커니즘 서술**이다.

> **We retrieve mechanisms, not events.**

## 5. 범위

- **Mechanism pool은 입력으로 주어진다.** 도메인별로 사전 구축된 두 종류의 자연어 지식 — 시점무관 **일반 원리(DK)** 와 사후 분석된 **역사적 사건(HE)** — JSON을 그대로 소비한다. 풀 생성 파이프라인은 별도 단계로, 본 작업 범위 밖.
- **본 작업의 초점**: 풀의 메커니즘 서술(`content`)을 현재 예측 맥락으로 검색하고, 검색 결과를 조건으로 하는 **메커니즘-인지 예측**. (검색 query를 어떻게 만드느냐는 이후 절에서 다룸.)

---

전체 방법론은 [state_conditioned_mechanism_retrieval_full_method.md](state_conditioned_mechanism_retrieval_full_method.md) 참조.
