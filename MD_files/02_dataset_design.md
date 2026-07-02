# 02. 데이터셋 디자인

EATF는 두 종류의 데이터를 쓴다.
- **(A) 예측 데이터** — Time-MMD 기반 수치+정렬 텍스트 (모델 입력·평가 대상)
- **(B) Mechanism Pool** — 검색 대상이 되는 사전 구축 자연어 지식 (별도 repo 산출물)

---

## A. 예측 데이터 (Time-MMD)

### 기반 데이터
**Time-MMD** (github.com/AdityaLab/Time-MMD) — 9개 도메인의 자연 발생 텍스트를 시계열과 fine-grained alignment. Fact/prediction 분리로 1차 contamination 방지.

### 도메인 (Monthly 5개)

| 도메인 | OT(타깃) 변수 |
|--------|---------|
| Agriculture | 가격 |
| Economy | 무역수지 |
| Security | 재난 보조금 |
| SocialGood | 실업률 |
| Traffic | 이동량 |

**제외**: Climate, Energy, Health (weekly), Environment (daily). Frequency 통일을 위해 daily/weekly를 monthly로 down-sample하면 짧은 이벤트가 묻히므로 monthly 5개만 사용.

### 시간 범위 — 텍스트 시대로 trim

각 도메인을 **텍스트가 존재하는 구간**으로 자른다. Time-MMD numerical은 텍스트보다 과거까지 확장돼 있어(예: SocialGood 1948~) 텍스트 이전 구간이 생기는데, 멀티모달 분석이 불가하므로 앞/뒤 텍스트-전무 구간만 제거한다.

| 도메인 | 텍스트 시대 범위 | 행 수 | (trim 전) |
|--------|------------------|-------|-----------|
| Agriculture | 1980-10 ~ 2024-04 | 523 | 532 |
| Economy | 1987-01 ~ 2024-03 | 447 | 447 |
| Security | 1998-09 ~ 2024-05 | 309 | 309 |
| SocialGood | 1980-01 ~ 2024-05 | 533 | 924 |
| Traffic | 1980-01 ~ 2024-03 | 531 | 651 |

- 시간순 split이라 시작점 통일 불필요. 내부의 산발적 no-info 월은 유지(sparsity일 뿐 시대 결손 아님).
- `build_dataset.py`의 `trim_to_text_coverage()` 구현.

### Look-back / Horizon / Split

- **L = 8** monthly (Time-MMD monthly 설정), **H = {6, 8, 10, 12}** monthly.
- **Split**: TSLib `Dataset_Custom` 표준 (train/val/test = 70/10/20, val·test는 직전 구간에서 look-back을 빌림). 작은 도메인(Security ~309행)도 모든 구간에 윈도우가 생기고, 예측 타깃은 각 구간 안에만 있어 누수 없음. 평가는 **각 도메인의 test split**에서 수행 (in-domain).

### 정규화

per-domain global StandardScaler (각 도메인 **train 행에만 fit**한 평균/분산으로 그 도메인 전체 표준화; MM-TSFlib/TSLib 표준). 이벤트(레짐 변화)를 도메인 기준 편차로 보존한다. **RevIN은 쓰지 않는다** — 각 윈도우를 look-back 통계로 정규화하면 이벤트 스파이크·레벨 변화를 지워 EATF 목적과 충돌.

### 다변량 변수 처리

다변량 도메인(Agriculture, Economy)은 변수별 커버리지가 다르다. 로더 `_trim_to_valid`가 실험별로 처리:
- **univariate (OT만)**: 텍스트 시대 풀시리즈.
- **multivariate (전 변수)**: 선택 변수가 모두 유효한 공통 윈도우.

> 공통 윈도우 trim은 OT의 정규화 std를 바꿔 절대 MSE를 변동시킨다(정규화 효과일 뿐, raw 오차는 동일). 주의.

### 텍스트 처리

Report + Search 둘 다 사용(Time-MMD 권장). **예외: Security는 search-only** — 원본 `Security_report.csv`가 전부 빈 껍데기라 report 모달리티 부재.

**병합**: 텍스트 윈도우 `[start_date,end_date]`를 majority-overlap으로 월에 배정(겹친 일수 많은 달, 동률시 빠른 월). 같은 월 다중 텍스트는 `"YYYY-MM-DD: text"`로 연결. Report/Search는 별도 컬럼. 빈 텍스트 row는 제외, 그 달에 유효 텍스트가 없으면 `"No information"`.

**최종 스키마**

| 컬럼 | 설명 |
|------|------|
| `date` | 월 시점 (YYYY-MM-01) |
| `start_date`, `end_date` | 원본 binary timestamp |
| `var1, ..., OT` | 다변량 시계열 값 (OT=타깃) |
| `report_text`, `search_text` | 그 월의 report/search fact 텍스트 |
| `report_pred`, `search_pred` | 그 월의 pred 텍스트 |

> **`*_pred` 컬럼은 모델 입력으로 쓰지 않는다** (미래 정보 누수). 현재 미사용.

### 파이프라인

```bash
uv run python build_dataset.py --domain Agriculture
uv run python build_dataset.py --all
```
도메인별 컬럼 차이는 `DOMAIN_CONFIG`의 `drop_cols`로 처리. 산출물 = `data/processed/{Domain}_merged.csv`.

---

## B. Mechanism Pool

검색 대상이 되는 자연어 지식 풀. **별도 repo `knowledge_pool`이 생성**하고, EATF는 그 JSON을 소비만 한다 (생성 파이프라인은 본 프로젝트 범위 밖).

### 두 풀 (도메인별, 완전 분리)

| 풀 | 성격 | content 스타일 | 규모 |
|---|---|---|---|
| **DK** (Domain Knowledge) | 시점무관 일반 원리·메커니즘 | 시점·고유명사 금지(이론명은 허용) | 도메인당 30~50 (saturate) |
| **HE** (Historical Event) | 특정 시점 사건 사후분석 | 시점·고유명사 필수, `abstract_summary`가 메커니즘화 요약 | unbounded, HE≫DK |

### 스키마

```json
{ "id": "...", "title": "...", "content": "자연어 2~6문장 (메커니즘 서술)",
  "evidence_docs": [{"doc_id": "...", "year": 2005, "snippet": "..."}] }
```
HE는 `date, date_precision, cause, effect, abstract_summary, quantitative_evidence` 추가. 구조화 필드 없음 — 정보는 전부 자연어 `content`에 두고, 이 `content`가 곧 retrieval 대상이다.

### 타깃과 cutoff

- 풀의 단위는 도메인이 아니라 **타깃 변수** (Economy=International Trade Balance 등). content엔 변수명을 박지 않되(범용성), 그 타깃에 영향을 주는 지식으로 한정.
- **DK는 cutoff 무관**(일반 원리는 시점 누설 아님). **HE는 평가 시 `max(evidence_docs.year) < test_time`으로 동적 필터링**해 누수 방지.

### 현황

pilot = Economy. `dk_pool` 140 entry, `he_pool` 18 entry. 나머지 도메인은 knowledge_pool에서 순차 구축 중.

---

## Memorization 처리 (robustness)

CiK 방식 — Setup-Clean(원본) vs Setup-Noise(Gaussian σ=3% 추가)의 성능 차이로 memorization 영향 정량화. 자세한 내용은 [04_evaluation.md](04_evaluation.md).
