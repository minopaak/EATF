# State-Conditioned Mechanism Retrieval for Multimodal Time-Series Forecasting

## 0. One-line Summary

본 방법은 기존 multimodal time-series forecasting의 **local text--time alignment**를 유지하면서, 전체 forecasting state에 대응되는 **global mechanism reasoning**을 검색·추론·인코딩하여 local representation과 결합하는 구조이다. 학습 과정에서는 validation forecasting error를 이용해 **state-to-mechanism query-writing prompt**를 반복적으로 개선한다.

---

## 1. 문제의식

기존 multimodal time-series forecasting은 보통 각 시점 또는 patch에 대응되는 text를 numerical sequence와 맞춘다.

\[
(x_t, e_t) \rightarrow z_t
\]

이 구조는 해당 시점에 어떤 textual evidence가 있었는지를 반영하는 데는 적합하다. 하지만 실제 미래 trajectory는 단순히 timestamp-level text만으로 결정되지 않는다. 현재 numerical-textual state 전체가 어떤 외부 mechanism에 의해 미래로 전이되는지가 중요할 수 있다.

예를 들어 다음 두 사건은 표면적으로 다르다.

```text
팬데믹
항만 폐쇄
노동 파업
전쟁으로 인한 물류 차질
```

하지만 forecasting 관점에서는 모두 다음과 같은 유사 mechanism을 공유할 수 있다.

```text
supply disruption → inventory shortage → delayed demand response → price/index change
```

반대로 표면적으로 비슷한 사건도 현재 state에 따라 다른 mechanism을 만들 수 있다.

```text
same event: demand recovery
case 1: low inventory → stockout risk → delayed supply response
case 2: high inventory → immediate sales recovery
case 3: high price level → suppressed demand recovery
```

따라서 필요한 것은 단순 text similarity나 event similarity가 아니라, **current forecasting state에 조건화된 mechanism similarity**이다.

---

## 2. 전체 Framework

전체 구조는 두 개의 branch와 하나의 fusion module로 구성된다.

```text
Local Branch:
Numerical Sequence + Textual Sequence
        ↓
Local text--time alignment
        ↓
Time-Series Encoder
        ↓
Local Fused Representation

Global Mechanism Branch:
Current Forecasting State
        ↓
State Summary
        ↓
State-to-Mechanism Query Writer
        ↓
Mechanism Retrieval
        ↓
Retrieved Mechanisms
        ↓
Reasoning Agent
        ↓
Mechanism Reasoning
        ↓
Mechanism Encoder
        ↓
Global Mechanism Representation

Fusion:
Local Fused Representation + Global Mechanism Representation
        ↓
Gated Cross-Attention Fusion
        ↓
Mechanism-aware Representation
        ↓
Forecasting Head
        ↓
Future Trajectory
```

수식으로는 다음과 같이 요약할 수 있다.

\[
\mathbf{Z}^{L} = f_{\text{local}}(\mathbf{X}, \mathbf{E})
\]

\[
\mathbf{Z}^{M} = f_{\text{mech}}(A_\psi(s_t, \mathrm{Retrieve}(Q_\rho(s_t), \mathcal{M})))
\]

\[
\mathbf{Z}^{F} = \mathrm{Fuse}(\mathbf{Z}^{L}, \mathbf{Z}^{M})
\]

\[
\hat{\mathbf{Y}}_{t+1:t+H} = \mathrm{Head}(\mathbf{Z}^{F})
\]

---

## 3. Input Formulation

### 3.1 Numerical Sequence

과거 numerical time-series window는 다음과 같다.

\[
\mathbf{X}_{t-L+1:t}
=
[\mathbf{x}_{t-L+1}, \dots, \mathbf{x}_{t}]
\in \mathbb{R}^{L \times C}
\]

여기서:

- \(L\): input window length
- \(C\): numerical variable 수
- \(\mathbf{x}_t\): time \(t\)의 numerical observation

예측 대상은 다음과 같다.

\[
\mathbf{Y}_{t+1:t+H}
=
[\mathbf{y}_{t+1}, \dots, \mathbf{y}_{t+H}]
\in \mathbb{R}^{H \times C_y}
\]

여기서:

- \(H\): forecast horizon
- \(C_y\): target variable 수

---

### 3.2 Textual Sequence

각 time step 또는 patch에 대응되는 textual input을 다음과 같이 둔다.

\[
\mathbf{E}_{t-L+1:t}
=
[e_{t-L+1}, \dots, e_t]
\]

각 \(e_i\)는 해당 시점의 textual evidence이다.

예시는 다음과 같다.

```text
news article summary
economic report
weather description
policy announcement
supply-chain note
domain-specific event description
internal operation log
```

기존 multimodal forecasting은 보통 다음 alignment를 수행한다.

\[
\mathbf{x}_i \leftrightarrow e_i
\]

이것이 **local text--time alignment**이다.

---

## 4. Local Text--Time Alignment Branch

### 4.1 Text Encoding

각 textual input \(e_i\)를 text encoder에 넣어 dense representation으로 변환한다.

\[
\mathbf{h}^{E}_i = f_{\text{text}}(e_i)
\]

전체 textual representation은 다음과 같다.

\[
\mathbf{H}^{E}
=
[\mathbf{h}^{E}_{t-L+1}, \dots, \mathbf{h}^{E}_{t}]
\in \mathbb{R}^{L \times d_e}
\]

사용 가능한 text encoder는 다음과 같다.

```text
frozen language model encoder
lightweight transformer text encoder
sentence embedding model
domain-specific text encoder
```

초기 구현에서는 frozen text encoder가 안전하다. 이 방법의 핵심은 text encoder 자체가 아니라 **mechanism retrieval과 global fusion**이기 때문이다.

---

### 4.2 Numerical Projection

Numerical sequence도 동일한 hidden dimension으로 projection한다.

\[
\mathbf{h}^{X}_i = \mathbf{W}_x \mathbf{x}_i + \mathbf{b}_x
\]

전체 numerical representation은 다음과 같다.

\[
\mathbf{H}^{X}
=
[\mathbf{h}^{X}_{t-L+1}, \dots, \mathbf{h}^{X}_{t}]
\in \mathbb{R}^{L \times d}
\]

---

### 4.3 Local Fusion

각 time step에서 numerical representation과 textual representation을 결합한다.

\[
\tilde{\mathbf{z}}^{L}_i
=
\phi
\left(
\mathbf{W}_{l}
[
\mathbf{h}^{X}_i ;
\mathbf{h}^{E}_i
]
+
\mathbf{b}_{l}
\right)
\]

여기서:

- \([\cdot ; \cdot]\): concatenation
- \(\phi\): activation function, 예: GELU
- \(\tilde{\mathbf{z}}^{L}_i \in \mathbb{R}^{d}\)

전체 local-aligned token sequence는 다음과 같다.

\[
\tilde{\mathbf{Z}}^{L}
=
[
\tilde{\mathbf{z}}^{L}_{t-L+1},
\dots,
\tilde{\mathbf{z}}^{L}_{t}
]
\in \mathbb{R}^{L \times d}
\]

이 단계가 그림에서 말하는 **Local Alignment**이다.

```text
Numerical Sequence:   x1      x2      x3      x4
                      |       |       |       |
Textual Sequence:     e1      e2      e3      e4

Result:
Local-aligned tokens: z1      z2      z3      z4
```

---

### 4.4 Time-Series Encoder

Local-aligned sequence \(\tilde{\mathbf{Z}}^{L}\)를 time-series encoder에 넣어 temporal dependency를 학습한다.

\[
\mathbf{Z}^{L}
=
f_{\text{ts}}(\tilde{\mathbf{Z}}^{L})
\]

여기서 \(f_{\text{ts}}\)는 다음 중 하나가 될 수 있다.

```text
Transformer encoder
PatchTST-style encoder
TSMixer
temporal convolution encoder
frozen time-series foundation model
lightweight forecasting backbone
```

출력은 다음과 같다.

\[
\mathbf{Z}^{L}
=
[
\mathbf{z}^{L}_1,
\dots,
\mathbf{z}^{L}_T
]
\in \mathbb{R}^{T \times d}
\]

여기서 \(T\)는 encoder 이후 token 수이다. Patch-based encoder를 사용하면 \(T\)는 patch 수가 될 수 있다.

이것이 **Local Fused Representation**이다.

---

## 5. Global Mechanism Branch

Local branch는 각 시점의 numerical-textual alignment를 학습한다. 그러나 본 논문에서 중요한 것은 별도의 global mechanism branch이다.

Global mechanism branch의 목표는 다음이다.

```text
(X, E)
    ↓
state summary
    ↓
mechanism query
    ↓
retrieved mechanisms
    ↓
mechanism reasoning
    ↓
mechanism representation
```

즉, mechanism representation은 raw text sequence를 그냥 다시 encoding해서 얻는 것이 아니다. 반드시 현재 forecasting state 전체를 기반으로 만들어야 한다.

---

## 6. Forecasting State Construction

현재 forecasting instance를 하나의 state로 정의한다.

\[
\mathcal{S}_t
=
(
\mathbf{X}_{t-L+1:t},
\mathbf{E}_{t-L+1:t},
\tau,
H
)
\]

여기서:

- \(\mathbf{X}_{t-L+1:t}\): historical numerical window
- \(\mathbf{E}_{t-L+1:t}\): aligned textual sequence
- \(\tau\): target specification
- \(H\): forecasting horizon

\(\tau\)에는 다음 정보가 들어갈 수 있다.

```text
target variable name
domain
unit
region
frequency
forecasting objective
```

예시:

```text
Target: trade index
Domain: macroeconomic forecasting
Region: United States
Frequency: monthly
Horizon: 6 months
```

이 state는 단일 timestep이 아니라, 전체 input window와 target setting을 포함한다.

---

## 7. State Summary

Mechanism retrieval을 위해 numerical-textual state를 retrieval 가능한 summary로 변환한다.

\[
s_t = \mathrm{Summarize}(\mathcal{S}_t)
\]

좋은 state summary에는 다음이 들어가야 한다.

```text
1. numerical trend
2. recent changes
3. abnormal patterns
4. salient textual events
5. target variable
6. forecast horizon
7. possible transition cues
```

예시:

```text
The trade index has shown a recent decline after a period of moderate growth.
The aligned textual evidence mentions rising oil prices, increased transportation costs,
and weakening external demand. The forecast target is the trade index over the next
six months.
```

이 summary는 이후 query writer의 입력이 된다.

---

## 8. State-to-Mechanism Query Writer

State summary를 mechanism retrieval query로 변환한다.

\[
q_t = Q_{\rho}(s_t)
\]

여기서 \(Q_{\rho}\)는 query writer이고, \(\rho\)는 query-writing rule 또는 prompt parameter이다.

중요한 점은 query가 단순 event keyword가 아니어야 한다는 것이다.

나쁜 query:

```text
oil price trade index
```

이건 surface event matching에 가깝다.

좋은 query:

```text
mechanisms by which rising oil prices affect trade activity through transportation costs,
import prices, and delayed demand contraction over a medium-term horizon
```

좋은 query는 다음 요소를 포함한다.

```text
current state
target variable
forecast horizon
external cue
intermediate pathway
possible delayed effect
domain-specific response channel
```

즉, query는 단순히 “무슨 사건이 있었는가?”를 묻는 것이 아니라 다음을 묻는다.

> 이 현재 state가 미래 target trajectory로 전이되는 작동 원리는 무엇인가?

---

## 9. Mechanism Pool

Mechanism pool은 raw document collection이 아니라, forecasting에 쓸 수 있는 mechanism-level knowledge의 집합이다.

\[
\mathcal{M} = \{m_i\}_{i=1}^{N}
\]

각 mechanism entry는 다음 구조를 가진다.

\[
m_i = (u_i, a_i, r_i, \mathbf{v}_i)
\]

여기서:

- \(u_i\): mechanism summary
- \(a_i\): source evidence
- \(r_i\): metadata
- \(\mathbf{v}_i\): embedding vector

예시:

```text
Mechanism summary:
Oil price shocks increase transportation and import costs, which can suppress trade
volume and induce delayed responses in trade-related indices.

Source evidence:
Economic report or historical case describing oil-price-driven trade contraction.

Metadata:
domain = macroeconomics
target = trade index
effect type = delayed negative effect
time scale = medium-term
```

단순 event summary와 mechanism summary는 다르다.

단순 event summary:

```text
Oil prices increased sharply.
```

Mechanism summary:

```text
Oil price increases raise transportation and import costs, which may reduce trade volume,
increase price pressure, and generate delayed changes in trade-related indicators.
```

Mechanism summary는 다음 구조를 가져야 한다.

\[
\text{cause}
\rightarrow
\text{intermediate pathway}
\rightarrow
\text{target effect}
\rightarrow
\text{temporal behavior}
\]

---

## 10. Mechanism Retrieval

Query \(q_t\)가 주어지면 mechanism pool에서 top-\(K\) mechanism을 retrieve한다.

\[
\mathcal{R}_t
=
\mathrm{TopK}
\left(
\mathrm{sim}
(
g(q_t),
\mathbf{v}_i
)
\right)
\]

여기서:

- \(g(\cdot)\): query embedding function
- \(\mathbf{v}_i\): mechanism embedding
- \(\mathrm{sim}\): cosine similarity 또는 dot product
- \(\mathcal{R}_t = \{m_{i_1}, \dots, m_{i_K}\}\): retrieved mechanisms

기존 retrieval과 차이는 다음과 같다.

```text
Existing retrieval:
state → relevant event/news

Proposed retrieval:
state → relevant mechanism
```

즉, 찾는 대상이 다르다.

---

## 11. Reasoning Agent

Retrieved mechanisms를 그대로 사용하는 것이 아니라, 현재 state와 함께 reasoning agent에 넣는다.

\[
r_t = A_{\psi}(s_t, \mathcal{R}_t)
\]

여기서:

- \(A_{\psi}\): reasoning agent
- \(s_t\): state summary
- \(\mathcal{R}_t\): retrieved mechanisms
- \(r_t\): mechanism reasoning text

Reasoning agent의 역할은 다음이다.

```text
1. retrieved mechanisms 중 현재 state에 적합한 것 선택
2. 현재 numerical-textual state와 mechanism 연결
3. future trajectory에 대한 direction 설명
4. horizon-specific effect 정리
5. 서로 충돌하는 mechanism이 있으면 정리
```

예시:

State summary:

```text
The trade index has recently declined after moderate growth.
Textual evidence mentions rising oil prices and weakening external demand.
The target is the trade index over the next six months.
```

Retrieved mechanism:

```text
Oil price shocks increase transportation and import costs, which can suppress trade
volume and induce delayed responses in trade-related indices.
```

Reasoning agent output:

```text
The current weakening trade index may be further affected by rising oil prices.
Higher oil prices can increase transportation and import costs, which may reduce trade
volume and create delayed downward pressure on the trade index. Since external demand
is also weakening, the negative effect may persist over the medium-term horizon rather
than appearing only as an immediate shock.
```

이 output이 **Mechanism Reasoning**이다.

---

## 12. Mechanism Encoder

Reasoning agent output \(r_t\)는 text 형태이므로, 이를 다시 encoder로 변환한다.

\[
\mathbf{Z}^{M} = f_{\text{mech}}(r_t)
\]

Token-level mechanism representation은 다음과 같다.

\[
\mathbf{Z}^{M}
=
[
\mathbf{z}^{M}_1,
\dots,
\mathbf{z}^{M}_{K_M}
]
\in \mathbb{R}^{K_M \times d}
\]

이 방식은 cross-attention fusion에 적합하다.

또는 pooled mechanism vector를 사용할 수도 있다.

\[
\mathbf{m} = \mathrm{Pool}(\mathbf{Z}^{M}) \in \mathbb{R}^{d}
\]

하지만 본 방법에서는 token-level representation을 기본으로 두고, ablation에서 pooled vector를 비교할 수 있다.

---

## 13. Local-Global Fusion

이제 두 representation이 준비되었다.

Local fused representation:

\[
\mathbf{Z}^{L} \in \mathbb{R}^{T \times d}
\]

Mechanism representation:

\[
\mathbf{Z}^{M} \in \mathbb{R}^{K_M \times d}
\]

이 둘을 결합하여 mechanism-aware representation을 만든다.

---

### 13.1 Fusion Direction

Attention에서 query는 local representation에서 나온다.

\[
Q = \mathbf{Z}^{L}\mathbf{W}_Q
\]

Key와 Value는 mechanism representation에서 나온다.

\[
K = \mathbf{Z}^{M}\mathbf{W}_K
\]

\[
V = \mathbf{Z}^{M}\mathbf{W}_V
\]

즉:

```text
Query: Local Fused Representation
Key: Mechanism Representation
Value: Mechanism Representation
```

이 방향이 중요한 이유는 최종적으로 업데이트되어야 하는 대상이 local temporal representation이기 때문이다.

즉, 각 local time token이 global mechanism representation을 참조한다.

```text
local token → attends to → mechanism tokens
```

---

### 13.2 Cross-Attention

Cross-attention은 다음과 같다.

\[
\mathbf{C}
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)V
\]

즉:

\[
\mathbf{C}
=
\mathrm{CrossAttn}
(
Q=\mathbf{Z}^{L},
K=\mathbf{Z}^{M},
V=\mathbf{Z}^{M}
)
\]

출력은 다음과 같다.

\[
\mathbf{C} \in \mathbb{R}^{T \times d}
\]

각 time token \(i\)에 대해:

\[
\mathbf{c}_i
=
\sum_{j=1}^{K_M}
\alpha_{ij}
\mathbf{v}^{M}_j
\]

여기서:

\[
\alpha_{ij}
=
\frac{
\exp
(
\mathbf{q}^{L}_i
\cdot
\mathbf{k}^{M}_j
/
\sqrt{d}
)
}{
\sum_{j'}
\exp
(
\mathbf{q}^{L}_i
\cdot
\mathbf{k}^{M}_{j'}
/
\sqrt{d}
)
}
\]

의미는 다음과 같다.

> 각 local time token이 mechanism reasoning 중 어떤 부분을 참고할지 선택한다.

---

### 13.3 Gated Fusion

Mechanism information은 항상 도움이 되는 것이 아니다. 검색된 mechanism이 부정확하거나 현재 state와 약하게 관련될 수도 있다.

따라서 gate를 둔다.

\[
\mathbf{G}
=
\sigma
(
\mathbf{W}_g
[
\mathbf{Z}^{L};
\mathbf{C}
]
+
\mathbf{b}_g
)
\]

여기서:

- \(\mathbf{G} \in \mathbb{R}^{T \times d}\)
- \(\sigma\): sigmoid
- \([\mathbf{Z}^{L};\mathbf{C}]\): local representation과 mechanism context의 concatenation

최종 mechanism-aware representation은 다음과 같다.

\[
\mathbf{Z}^{F}
=
\mathrm{LN}
\left(
\mathbf{Z}^{L}
+
\mathbf{G}
\odot
\mathbf{C}
\right)
\]

여기서:

- \(\mathbf{Z}^{L}\): 기존 local fused representation
- \(\mathbf{C}\): mechanism-aware context
- \(\mathbf{G}\): mechanism 반영 정도
- \(\odot\): element-wise multiplication
- \(\mathrm{LN}\): layer normalization

이 구조의 의미는 다음과 같다.

> local representation을 기본으로 유지하되, 필요한 경우에만 mechanism context를 선택적으로 주입한다.

---

## 14. Forecasting Head

Fusion 이후에는 mechanism-aware representation이 생성된다.

\[
\mathbf{Z}^{F} \in \mathbb{R}^{T \times d}
\]

여기에 forecasting head를 붙여 미래 trajectory를 예측한다.

\[
\hat{\mathbf{Y}}_{t+1:t+H}
=
\mathrm{Head}(\mathbf{Z}^{F})
\]

가장 단순한 방식은 flatten 후 MLP를 적용하는 것이다.

\[
\hat{\mathbf{Y}}
=
\mathrm{MLP}
(
\mathrm{Flatten}(\mathbf{Z}^{F})
)
\]

출력 차원은 다음과 같다.

\[
\hat{\mathbf{Y}} \in \mathbb{R}^{H \times C_y}
\]

---

## 15. Training Objective

기본 forecasting loss는 다음과 같다.

\[
\mathcal{L}_{\text{pred}}
=
\frac{1}{H C_y}
\sum_{h=1}^{H}
\sum_{c=1}^{C_y}
\ell
(
\hat{y}_{t+h,c},
y_{t+h,c}
)
\]

\(\ell\)은 다음 중 하나를 사용할 수 있다.

```text
MSE
MAE
Huber loss
sMAPE-based loss
```

안정성을 위해 Huber loss를 사용할 수 있다.

\[
\ell_{\delta}(a)
=
\begin{cases}
\frac{1}{2}a^2, & |a| \leq \delta \\
\delta(|a| - \frac{1}{2}\delta), & |a| > \delta
\end{cases}
\]

\[
a = \hat{y} - y
\]

이 loss로 다음 neural modules를 학습한다.

```text
numerical projection
local fusion module
time-series encoder
gated cross-attention fusion
forecasting head
```

초기 버전에서는 다음을 frozen으로 둘 수 있다.

```text
text encoder
mechanism text encoder
reasoning LLM
retrieval encoder
```

---

## 16. 학습 과정: Inner Loop와 Outer Loop

학습 과정은 두 층으로 나뉜다.

```text
Inner loop:
neural forecasting model 학습

Outer loop:
query-writing prompt / rule 개선
```

즉, 모든 것을 end-to-end gradient로 학습하는 것이 아니다.

Neural modules는 forecasting loss로 학습하고, query-writing prompt는 validation forecasting error를 기반으로 reflection을 통해 개선한다.

---

## 17. Inner Loop: Neural Model Training

### 17.1 고정된 query prompt로 retrieval 수행

초기 query-writing prompt \(\rho^{(0)}\)를 둔다.

이 prompt를 사용해 각 training instance에서 query를 생성한다.

\[
q_t^{(0)} = Q_{\rho^{(0)}}(s_t)
\]

그 다음 mechanism retrieval을 수행한다.

\[
\mathcal{R}_t^{(0)} = \mathrm{Retrieve}(q_t^{(0)}, \mathcal{M})
\]

Reasoning agent가 mechanism reasoning을 만든다.

\[
r_t^{(0)} = A_{\psi}(s_t, \mathcal{R}_t^{(0)})
\]

Mechanism encoder가 이를 representation으로 바꾼다.

\[
\mathbf{Z}^{M}_t = f_{\text{mech}}(r_t^{(0)})
\]

Local branch는 다음을 만든다.

\[
\mathbf{Z}^{L}_t = f_{\text{local}}(\mathbf{X}_t, \mathbf{E}_t)
\]

Fusion 후 예측한다.

\[
\mathbf{Z}^{F}_t
=
\mathrm{GatedCrossAttn}
(
\mathbf{Z}^{L}_t,
\mathbf{Z}^{M}_t
)
\]

\[
\hat{\mathbf{Y}}_t
=
\mathrm{Head}(\mathbf{Z}^{F}_t)
\]

---

### 17.2 Neural Model Update

각 mini-batch에 대해:

```text
1. local fused representation 생성
2. cached mechanism reasoning 또는 on-the-fly reasoning 사용
3. mechanism representation 생성
4. gated cross-attention fusion 수행
5. forecasting head로 예측
6. forecasting loss 계산
7. neural parameters update
```

Gradient update 대상은 다음이다.

```text
numerical projection
local fusion layer
time-series encoder
cross-attention fusion module
gate module
forecasting head
```

Query prompt, LLM reasoning agent, mechanism pool은 inner loop에서 gradient로 업데이트하지 않는다.

---

## 18. Outer Loop: Forecast-Aware Query Prompt Adaptation

여기서 중요한 부분은 **검색 쿼리 작성 prompt가 학습 과정에서 개선된다**는 점이다.

단, 이것은 gradient descent로 prompt embedding을 학습한다는 뜻이 아니다. Validation forecasting error를 보고 query-writing rule 또는 prompt를 LLM reflection으로 업데이트하는 방식이다.

전체 흐름은 다음과 같다.

```text
1. 현재 query prompt로 train/validation instance의 query 생성
2. mechanism retrieval 수행
3. reasoning agent가 mechanism reasoning 생성
4. forecasting model이 예측 수행
5. validation error 계산
6. high-error cases 수집
7. error 원인 분석
8. query-writing prompt 수정
9. 수정된 prompt로 retrieval cache 재생성
10. neural model 재학습 또는 fine-tuning
```

---

## 19. Round-Based Prompt Adaptation

초기 prompt를 \(\rho^{(0)}\)라고 하자.

각 round \(r\)에서 다음을 수행한다.

\[
q_t^{(r)} = Q_{\rho^{(r)}}(s_t)
\]

\[
\mathcal{R}_t^{(r)} = \mathrm{Retrieve}(q_t^{(r)}, \mathcal{M})
\]

\[
r_t^{(r)} = A_{\psi}(s_t, \mathcal{R}_t^{(r)})
\]

\[
\hat{\mathbf{Y}}_t^{(r)}
=
F_{\theta}^{(r)}
(
\mathbf{X}_t,
\mathbf{E}_t,
r_t^{(r)}
)
\]

Validation error는 다음과 같다.

\[
e_t^{(r)}
=
\mathcal{L}
(
\hat{\mathbf{Y}}_t^{(r)},
\mathbf{Y}_t
)
\]

오차가 큰 case를 모은다.

\[
\mathcal{H}^{(r)}
=
\mathrm{TopErrorCases}
(
\{e_t^{(r)}\}
)
\]

그다음 reflection module이 prompt를 업데이트한다.

\[
\rho^{(r+1)}
=
\mathrm{Reflect}
(
\rho^{(r)},
\mathcal{H}^{(r)}
)
\]

중요한 점은 reflection 대상이다.

기존 event-based method에서는 event filtering이나 reasoning trace를 개선할 수 있다. 그러나 본 방법에서 reflection은 **state를 mechanism query로 변환하는 rule**을 개선한다.

즉, adaptation target은 다음이다.

```text
state-to-mechanism query rule
```

---

## 20. Query Prompt 개선이 필요한 이유

Mechanism retrieval은 query에 매우 민감하다.

예를 들어 초기 query prompt가 너무 단순하면 다음과 같은 query가 나온다.

```text
oil price trade index decline
```

이 query는 표면적으로 oil price와 trade index가 같이 나오는 문서를 잘 찾을 수 있다.

하지만 forecasting에 필요한 것은 단순 동시출현 문서가 아니라, 다음과 같은 mechanism이다.

```text
oil price shock → transportation cost increase
→ import/export cost pressure
→ delayed contraction in trade activity
→ trade index decline
```

따라서 query prompt는 다음 방향으로 개선되어야 한다.

```text
event keyword 중심
→ mechanism pathway 중심

surface similarity 중심
→ transition mechanism 중심

current text 중심
→ current state + target + horizon 중심
```

---

## 21. 초기 Query-Writing Prompt 예시

초기 query-writing prompt는 다음과 같이 둘 수 있다.

```text
You are given a forecasting state consisting of a numerical trend summary,
aligned textual evidence, target variable, and forecast horizon.

Write a search query to retrieve external mechanisms that may explain how
the current state can evolve into the future target trajectory.

The query should include:
- the target variable
- the salient numerical pattern
- the relevant external textual cues
- the forecast horizon
- possible mechanism pathways
```

State summary 예시:

```text
The trade index has recently declined. Textual evidence mentions rising oil prices,
higher transportation costs, and weakening external demand. The target horizon is six months.
```

Generated query 예시:

```text
mechanisms by which rising oil prices and weakening external demand affect trade index
over a six-month horizon through transportation costs and trade volume contraction
```

---

## 22. High-Error Case 분석 예시

### 22.1 Case A: query가 surface event에 치우친 경우

Validation case:

```text
Numerical pattern:
The demand index remains flat but begins to decline slightly near the end of the window.

Textual evidence:
Several reports mention port congestion and shipping delays.

Target:
Retail demand index over the next four weeks.
```

초기 query:

```text
port congestion shipping delay retail demand
```

Retrieved mechanisms:

```text
1. Port congestion causes delivery delays.
2. Shipping delays affect logistics performance.
3. Retailers experience slower product delivery.
```

예측 결과:

```text
Model predicts an immediate sharp decline.
```

실제 결과:

```text
Demand remains stable for two weeks and declines later.
```

오차 원인:

```text
Retrieved mechanisms captured the event type but missed the delayed inventory-mediated pathway.
The query did not include current inventory or delayed demand response.
```

Reflection 결과:

```text
When logistics-related events appear, the query should include possible intermediate
mechanisms such as inventory buffers, delayed stockout, substitution, and lagged demand response.
```

업데이트된 query prompt rule:

```text
For supply-chain or logistics events, explicitly ask for mechanisms involving
inventory buffers, delayed stockout, substitution effects, and lagged demand response,
rather than only searching for the event itself.
```

업데이트 후 query:

```text
mechanisms by which port congestion affects retail demand through inventory buffers,
delayed stockouts, substitution behavior, and lagged demand response over a four-week horizon
```

---

### 22.2 Case B: query가 target variable을 충분히 반영하지 못한 경우

Validation case:

```text
Numerical pattern:
Electricity demand shows a mild upward trend.

Textual evidence:
News reports mention a heatwave.

Target:
Electricity load over the next 7 days.
```

초기 query:

```text
heatwave future demand
```

Retrieved mechanisms:

```text
1. Heatwaves affect public health.
2. Heatwaves increase general cooling needs.
3. Heatwaves disrupt outdoor activity.
```

예측 결과:

```text
Model predicts only a small increase.
```

실제 결과:

```text
Electricity load increases sharply.
```

오차 원인:

```text
The query did not specify electricity load as the target variable.
Retrieved mechanisms were too broad and not load-specific.
```

Reflection 결과:

```text
Queries must explicitly include the target variable and domain-specific response channel.
For electricity load, heatwave queries should include cooling demand, air conditioning use,
peak load, and short-term demand surge.
```

업데이트된 query:

```text
mechanisms by which heatwaves increase electricity load through cooling demand,
air conditioning usage, and peak load surge over a 7-day horizon
```

---

### 22.3 Case C: query가 horizon을 반영하지 못한 경우

Validation case:

```text
Numerical pattern:
Trade index is stable.

Textual evidence:
Reports mention newly announced tariffs.

Target:
Trade index over the next 12 months.
```

초기 query:

```text
tariff trade index effect
```

Retrieved mechanisms:

```text
1. Tariffs affect import costs.
2. Tariffs reduce trade volume.
```

예측 결과:

```text
Model predicts an immediate drop.
```

실제 결과:

```text
The effect appears gradually after several months.
```

오차 원인:

```text
The query retrieved general tariff mechanisms but did not focus on long-horizon delayed adjustment.
```

Prompt update:

```text
When the forecast horizon is medium- or long-term, the query should explicitly ask for
lag structure, delayed adjustment, contract renewal effects, and gradual pass-through.
```

업데이트된 query:

```text
long-term mechanisms by which newly announced tariffs affect trade index through
import cost pass-through, contract renewal, delayed substitution, and gradual trade volume adjustment
```

---

## 23. Reflection Prompt 예시

Query prompt를 개선하기 위한 reflection prompt는 다음과 같이 설계할 수 있다.

```text
You are improving a query-writing prompt for mechanism retrieval in multimodal
time-series forecasting.

You are given high-error validation cases. Each case includes:
1. state summary
2. generated query
3. retrieved mechanisms
4. mechanism reasoning
5. model prediction
6. ground-truth future trajectory
7. error pattern

Analyze why the query failed to retrieve useful mechanisms.

Focus on:
- whether the query was too event-specific
- whether it missed the target variable
- whether it missed the forecast horizon
- whether it ignored the numerical trend
- whether it failed to include intermediate pathways
- whether it retrieved surface-similar but mechanism-irrelevant entries

Then revise the query-writing prompt so future queries better retrieve
mechanism-level knowledge conditioned on the forecasting state.
```

Reflection output 예시:

```text
Observed failure:
The current query prompt often produces event-keyword queries. These queries retrieve
documents about the same event type but fail to capture the mechanism linking the current
state to the future target trajectory.

Prompt revision:
The query writer must explicitly include:
1. target variable
2. forecast horizon
3. observed numerical trend
4. salient textual cues
5. intermediate causal pathway
6. possible lagged or delayed effect
7. domain-specific response channel

Updated instruction:
Do not write a query that only lists event keywords. Write a mechanism-seeking query
that asks how the current state may transition into the future target trajectory.
```

---

## 24. Query-Writing Prompt 업데이트 전후

### Before

```text
Write a search query using the current text and target variable.
```

생성 query:

```text
oil price trade index
```

### After

```text
Write a mechanism-seeking retrieval query for time-series forecasting.

The query must not merely list event keywords. It should ask for mechanisms that explain
how the current numerical-textual state may evolve into the future target trajectory.

Include:
- target variable
- forecast horizon
- observed numerical trend
- salient textual cues
- intermediate pathways
- possible delayed or lagged effects
- domain-specific response channel
```

생성 query:

```text
mechanisms by which rising oil prices under a weakening trade-index state affect
future trade activity through transportation costs, import cost pass-through,
reduced trade volume, and delayed medium-term adjustment
```

이렇게 prompt가 개선되면서 retrieval target이 event에서 mechanism으로 이동한다.

---

## 25. Prompt Adaptation Algorithm

논문에는 다음 algorithm 형태로 넣을 수 있다.

```text
Algorithm: Forecast-Aware Query Prompt Adaptation

Input:
Training set D_train
Validation set D_val
Mechanism pool M
Initial query-writing prompt rho_0
Number of adaptation rounds R

for r = 0, ..., R-1 do:

    1. Query generation
       For each instance in D_train and D_val:
           summarize forecasting state s_t
           generate mechanism query q_t using prompt rho_r

    2. Mechanism retrieval
       retrieve top-K mechanisms R_t from mechanism pool M

    3. Mechanism reasoning
       generate reasoning text r_t using state summary and retrieved mechanisms

    4. Neural model training
       train forecasting model F_theta using D_train and retrieved/reasoned mechanisms

    5. Validation
       evaluate F_theta on D_val
       compute forecasting errors e_t

    6. Error case selection
       select high-error cases H_r

    7. Reflection
       analyze whether failures come from poor query formulation,
       irrelevant retrieval, missing horizon, missing target variable,
       or missing mechanism pathway

    8. Prompt update
       revise query-writing prompt:
           rho_{r+1} = Reflect(rho_r, H_r)

    9. Retrieval cache refresh
       regenerate queries and retrieved mechanisms using rho_{r+1}

end for

Return:
final query-writing prompt rho_R
trained forecasting model F_theta
```

---

## 26. Training Stage 구분

실제 구현에서는 세 단계로 나누는 것이 좋다.

### Stage 1: Mechanism Pool Construction

Offline 단계이다.

```text
raw documents / historical cases
        ↓
mechanism extraction prompt
        ↓
mechanism summaries
        ↓
embedding
        ↓
mechanism pool
```

여기서는 forecasting model을 학습하지 않는다. 외부 mechanism knowledge base를 만드는 단계이다.

---

### Stage 2: Prompt Adaptation + Model Training

Train/validation set을 사용한다.

```text
current prompt
        ↓
query generation
        ↓
mechanism retrieval
        ↓
reasoning
        ↓
forecasting model training
        ↓
validation error
        ↓
prompt reflection
        ↓
updated prompt
```

이 단계에서 query-writing prompt가 개선된다.

---

### Stage 3: Test Inference

Test 단계에서는 prompt를 더 이상 수정하지 않는다.

```text
fixed query-writing prompt
        ↓
state summary
        ↓
query generation
        ↓
mechanism retrieval
        ↓
reasoning
        ↓
mechanism-aware forecasting
```

중요한 점은 다음이다.

> Test set의 ground truth를 사용해 prompt를 수정하면 안 된다.

즉, prompt adaptation은 train/validation에서만 수행하고, test에서는 고정된 prompt를 사용한다.

---

## 27. 학습 과정 전체 예시

### 27.1 Round 0

초기 query prompt:

```text
Generate a search query using the textual evidence and target variable.
```

State summary:

```text
The trade index has recently declined. Textual evidence mentions rising oil prices,
transportation cost increases, and weakening external demand. The forecast horizon is six months.
```

Generated query:

```text
oil price trade index external demand
```

Retrieved mechanisms:

```text
1. Oil prices affect global markets.
2. External demand affects trade.
3. Energy prices are related to transportation costs.
```

Reasoning:

```text
Oil prices and external demand may affect the trade index.
```

문제:

```text
The retrieved mechanisms are too general.
The reasoning does not specify the transition pathway.
The query lacks lag structure and target-specific mechanism.
```

Validation 결과:

```text
Prediction: mild decline
Ground truth: delayed but stronger decline
Error: high
```

---

### 27.2 Reflection after Round 0

Reflection 분석:

```text
The query prompt tends to produce keyword-based queries.
It does not force the query to include intermediate pathways or delayed effects.
For macroeconomic indices, queries should include pass-through, lagged adjustment,
and target-specific response channels.
```

Updated prompt:

```text
Write a mechanism-seeking query, not an event-keyword query.
The query must include the target variable, forecast horizon, observed numerical trend,
salient textual cues, intermediate pathways, and possible delayed effects.
```

---

### 27.3 Round 1

Updated generated query:

```text
mechanisms by which rising oil prices and weakening external demand affect the trade index
through transportation cost pass-through, reduced trade volume, and delayed medium-term adjustment
over a six-month horizon
```

Retrieved mechanisms:

```text
1. Oil price shocks raise transportation and import costs, reducing trade volume with lag.
2. External demand weakening suppresses export activity and lowers trade-related indices.
3. Cost pass-through can create delayed adjustment in macroeconomic indicators.
```

Reasoning:

```text
The current decline in the trade index is likely to be amplified by rising oil prices
and weakening external demand. Oil prices increase transportation and import costs,
which may reduce trade volume after a delay. Since the horizon is six months, the
effect is expected to emerge gradually rather than immediately.
```

Validation 결과:

```text
Prediction: gradual medium-term decline
Ground truth: gradual medium-term decline
Error: reduced
```

이렇게 prompt가 개선되면서 retrieval quality가 좋아지고, mechanism reasoning도 forecasting에 더 적합해진다.

---

## 28. Prompt 개선은 무엇을 학습하는가?

이 방법에서 prompt adaptation은 다음을 학습한다.

```text
어떤 state summary에서
어떤 kind of mechanism query를 만들어야
forecasting에 도움이 되는 mechanism을 retrieve할 수 있는가
```

즉, 학습되는 것은 단순 문장 템플릿이 아니라 다음 mapping이다.

```text
state → mechanism query로 변환하는 규칙
```

초기에는 query가 이런 식이다.

```text
event + target keyword
```

개선 후에는 이런 식이 된다.

```text
state + target + horizon + pathway + lag effect + domain response channel
```

논문에서의 claim은 다음과 같이 잡을 수 있다.

> We do not adapt the retrieval corpus or train a dense retriever directly. Instead, we adapt the state-to-mechanism query-writing rule using forecast-aware validation feedback.

한국어로는 다음과 같다.

> 우리는 retrieval corpus나 dense retriever 자체를 학습시키는 것이 아니라, forecasting validation error를 이용해 현재 state를 mechanism-seeking query로 변환하는 query-writing rule을 개선한다.

---

## 29. Ablation Study 제안

### 29.1 Without Mechanism Branch

\[
\mathbf{Z}^{F} = \mathbf{Z}^{L}
\]

Local fused representation만 사용한다.

목적:

```text
mechanism branch가 실제로 성능을 높이는지 확인
```

---

### 29.2 Raw Retrieved Text instead of Mechanism Reasoning

Retrieved mechanism을 reasoning agent 없이 바로 encoding한다.

\[
\mathbf{Z}^{M} = f_{\text{mech}}(\mathcal{R}_t)
\]

목적:

```text
reasoning agent가 필요한지 확인
```

---

### 29.3 Event Retrieval instead of Mechanism Retrieval

Mechanism pool 대신 event/news pool을 retrieve한다.

```text
state → event retrieval
```

목적:

```text
retrieval target이 event가 아니라 mechanism이어야 한다는 점 검증
```

---

### 29.4 Concatenation Fusion

Gated cross-attention 대신 concat fusion을 사용한다.

\[
\mathbf{Z}^{F}
=
\mathrm{MLP}
([
\mathbf{Z}^{L};
\mathbf{m}
])
\]

목적:

```text
fusion mechanism의 중요성 검증
```

---

### 29.5 Cross-Attention without Gate

\[
\mathbf{Z}^{F}
=
\mathrm{LN}
(
\mathbf{Z}^{L}
+
\mathrm{CrossAttn}
(
\mathbf{Z}^{L},
\mathbf{Z}^{M}
)
)
\]

목적:

```text
gate가 noisy mechanism을 조절하는 데 필요한지 확인
```

---

### 29.6 Without Forecast-Aware Query Adaptation

초기 query rule만 사용하고 validation feedback으로 갱신하지 않는다.

목적:

```text
query adaptation의 효과 검증
```

---

## 30. 전체 Method Summary

최종 method는 다음과 같다.

```text
1. Numerical sequence와 textual sequence를 입력받는다.
2. Textual sequence는 text encoder로 encoding한다.
3. Numerical representation과 textual representation을 time step별로 local fusion한다.
4. Time-series encoder를 통해 local fused representation을 만든다.
5. 전체 numerical-textual state를 summary로 변환한다.
6. State summary를 mechanism-seeking query로 바꾼다.
7. Mechanism pool에서 query와 관련 있는 mechanism을 retrieve한다.
8. Reasoning agent가 current state와 retrieved mechanisms를 연결해 mechanism reasoning을 만든다.
9. Mechanism reasoning을 encoder에 넣어 global mechanism representation을 만든다.
10. Local fused representation을 query로, mechanism representation을 key/value로 하는 gated cross-attention fusion을 수행한다.
11. Mechanism-aware representation에 forecasting head를 적용해 미래 trajectory를 예측한다.
12. 학습 중 validation high-error cases를 분석하여 query-writing prompt를 개선한다.
13. 개선된 prompt로 retrieval을 다시 수행하고, model을 재학습 또는 fine-tuning한다.
14. Test에서는 최종 prompt와 model을 고정하고 inference만 수행한다.
```

---

## 31. 논문용 핵심 문장

Method section에 들어갈 문장은 다음과 같이 쓸 수 있다.

```text
The proposed framework consists of a local alignment branch and a global mechanism branch.
The local branch aligns numerical observations with co-occurring textual inputs at the
time-step or patch level, producing a local fused representation. The global branch first
summarizes the entire forecasting state and converts it into a mechanism-seeking query.
The query retrieves relevant mechanism summaries from an external mechanism pool, and a
reasoning agent contextualizes the retrieved mechanisms with respect to the current state.
The resulting mechanism reasoning is encoded as a global mechanism representation.
Finally, the local representation queries the mechanism representation through gated
cross-attention, producing a mechanism-aware representation for forecasting.
```

Prompt adaptation 문장은 다음과 같이 쓸 수 있다.

```text
To improve mechanism retrieval, we further introduce forecast-aware query prompt adaptation.
After each training round, we evaluate the model on validation instances and collect
high-error cases. A reflection module analyzes whether the generated queries failed to
include the target variable, forecast horizon, numerical trend, intermediate pathway, or
lagged effect. Based on this analysis, the query-writing prompt is revised and the retrieval
cache is regenerated. This process improves the state-to-mechanism query rule using
forecasting feedback, without using test labels or directly training the retriever.
```
