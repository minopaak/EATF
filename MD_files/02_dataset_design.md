# 02. 데이터셋 디자인

## 스코프

### 기반 데이터
**Time-MMD** (github.com/AdityaLab/Time-MMD)

선정 이유:
- 9개 도메인의 자연 발생 텍스트를 시계열과 fine-grained alignment
- Fact/prediction 분리로 1차 contamination 방지 완료
- Event 차원만 추가하면 우리 contribution 명확

### 도메인 (Monthly 5개)

| 도메인 | OT 변수 |
|--------|---------|
| Agriculture | 가격 |
| Economy | 무역수지 |
| Security | 재난 보조금 |
| SocialGood | 실업률 |
| Traffic | 이동량 |

**제외**: Climate, Energy, Health (weekly), Environment (daily). Future work.

**제외 이유**: Frequency 통일 시 daily/weekly를 monthly로 down-sample하면 짧은 이벤트의 ROI가 묻혀버림. Event-aware 데이터셋의 핵심 가치 훼손. Monthly 5개만 사용하면 정보 손실 없음.

> **Note**: 초기 설계에서 Climate를 monthly로 분류했으나 실제 Time-MMD 데이터는 7일 간격 weekly이므로 제외함 (Phase 1 데이터 검증 단계에서 발견).

### 시간 범위 — 텍스트 시대로 trim

이벤트-aware 멀티모달 벤치마크이므로, 각 도메인을 **텍스트(report 또는 search)가 존재하는 구간**으로 자른다. Time-MMD numerical은 텍스트보다 과거까지 확장돼 있어(예: SocialGood 1948~, Traffic 1970~) 텍스트 시대 이전 구간이 생기는데, 그 구간은 멀티모달 분석이 불가하므로 제거한다.

| 도메인 | 텍스트 시대 범위 | 행 수 | (trim 전) |
|--------|------------------|-------|-----------|
| Agriculture | 1980-10 ~ 2024-04 | 523 | 532 |
| Economy | 1987-01 ~ 2024-03 | 447 | 447 (변화 없음) |
| Security | 1998-09 ~ 2024-05 | 309 | 309 (변화 없음) |
| SocialGood | 1980-01 ~ 2024-05 | 533 | 924 |
| Traffic | 1980-01 ~ 2024-03 | 531 | 651 |

- 시간순 split이라 시작점 통일 불필요.
- **내부의 산발적 no-info 월은 유지** (그건 sparsity일 뿐 시대 결손이 아님). trim은 앞/뒤 텍스트-전무 구간만 제거.
- `build_dataset.py`의 `trim_to_text_coverage()` 구현.

> **MM-TSFlib과의 관계**: MM-TSFlib은 텍스트 기준으로 자르지 **않는다** (텍스트 없는 행도 보유, 예: SocialGood 1950년대). 그들의 더 짧은 범위는 단지 더 오래된 numerical 데이터 vintage 때문 — "MM 맞추기"는 의미 없고, 텍스트-커버리지 trim은 우리 고유의 원칙적 선택이다. (Traffic은 우연히 둘 다 1980 시작이라 531행으로 일치.)

### Look-back / Horizon

- L = 8 monthly — Time-MMD 논문의 monthly 설정 (daily=96, weekly=36, monthly=8)
- H = {6, 8, 10, 12} monthly — Time-MMD monthly horizons

> **Split**: TSLib `Dataset_Custom` 표준 방식 (train/val/test = 70/10/20, val·test는 직전 구간에서 look-back을 빌림). 작은 도메인(Security ~309행)도 모든 구간에 윈도우가 생김. 예측 타깃은 각 구간 안에만 있어 누수 없음. 평가는 **target의 test split**에서 수행 (in-domain·cross-domain 동일 셋). cross-domain의 정확한 설계는 `04_evaluation.md` 참조 — 단순 TS-swap이 아니라 multimodal 텍스트/이벤트 transfer.

### 다변량 변수 처리

다변량 도메인(Agriculture, Economy)은 변수별 커버리지가 다름. 예: Agriculture의 OT는 1980~2024 완전하지만 보조 변수(Wholesale, Spread)는 1990~2023만 존재.

**정책: 변수 trim은 build_dataset이 아니라 로더가 실험별로 처리** (build_dataset은 위의 텍스트-시대 trim만 하고 보조변수 초기 NaN은 그대로 둔다). 로더 `_trim_to_valid`가 선택된 변수들이 모두 유효한 연속 구간으로 자른다:
- **univariate (OT만)**: 텍스트 시대 풀시리즈 (보조변수 무시 → 추가 trim 거의 없음). cross-domain LODO의 기본.
- **multivariate (전 변수)**: 공통 윈도우 (보조변수 유효구간). 예: Agriculture는 Wholesale/Spread가 1990~이라 1990~2023으로 좁혀짐.

> **중요 (정규화 주의)**: 공통 윈도우 trim은 OT의 정규화 std를 바꿔 **절대 MSE를 변동**시킨다(예: Agri OT 단변량 0.23 → 다변량 공통윈도우 0.58). 이는 예측 실력 저하가 아니라 정규화 효과일 뿐 — raw 오차는 동일(×0.98~1.15)함을 검증함. degradation **비율**(cross/in-domain)에서는 target std가 약분돼 영향 없음. RevIN으로 바꾸는 건 안 됨 (이벤트 스파이크/레벨 변화를 정규화가 지워버려 우리 목적과 정면충돌). 자세한 검증은 `04_evaluation.md` / 메모리 참조.

## 텍스트 처리

### 기본 원칙
Report + Search 둘 다 사용. Time-MMD 저자 권장 방식이고 기존 multimodal 연구들 표준.

> **예외: Security는 search-only.** Time-MMD 원본 `Security_report.csv`는 19,569 row 전부 `fact`/`preds`가 NaN인 빈 껍데기. report 모달리티가 통째로 없음 → Security report_text/report_pred는 전부 `"No information"`. multimodal 평가 시 Security는 search 텍스트만 유효하게 취급.

### 통합 방식

**병합 단계**:
1. 텍스트 row를 majority-overlap 정책으로 월에 배정 (위 "텍스트-월 매칭 정책" 참조)
2. 같은 월에 여러 텍스트 → `"YYYY-MM-DD: text"` 형식으로 줄바꿈 연결
3. Report와 Search는 **별도 컬럼**으로 분리
4. Pred 정보도 별도 컬럼으로 (라벨링에 활용)
5. **내용이 빈 텍스트 row(fact/preds가 NaN)는 제외** — Time-MMD에 빈 row 다수 존재 (Security report 전체, search의 결과 없는 주 등). 빈 prefix(`"날짜: "`) artifact 방지
6. 그 달에 유효 텍스트가 하나도 없으면: `"No information"`

> **텍스트 sparsity (실측)**: report no-info 22.8%~**100%**(Security), search no-info 0%~41.9%. 도메인별 편차 큼 → cross-domain transfer 시 sparsity gap이 noise 요인. EDA 노트북 참조.

### 최종 스키마

| 컬럼 | 설명 |
|------|------|
| `date` | 월 시점 (YYYY-MM-01) |
| `start_date`, `end_date` | 원본 binary timestamp |
| `var1, var2, ..., OT` | 다변량 시계열 값 |
| `report_text` | 그 월의 report fact 텍스트 |
| `search_text` | 그 월의 search fact 텍스트 |
| `report_pred` | 그 월의 report pred 텍스트 |
| `search_pred` | 그 월의 search pred 텍스트 |

### Pred 활용 정책

`*_pred` 컬럼은 **모델 입력으로 절대 사용 안 함** (causal leakage). 라벨링 단계에서만 활용:
- ROI 길이 추정 (예: "next 6 months" 같은 시간 표현 추출)
- 영향 방향 추정 (positive/negative)

자세한 내용은 `03_roi_annotation.md` 참조.

## 데이터 통합 파이프라인

### 구현
`build_dataset.py` 참조. 사용법:
```
uv run python build_dataset.py --domain Agriculture
uv run python build_dataset.py --all
```

도메인별 차이는 `DOMAIN_CONFIG` dict의 `drop_cols`로 처리 (중복 date, 메타데이터 컬럼).

### 검증 항목 (Phase 1 완료)
- [x] Numerical CSV 컬럼명: 도메인별 차이 있음 → per-domain `drop_cols`로 정규화 (Agriculture/Traffic의 `Date`, Economy의 `Month` 제거)
- [x] 텍스트 CSV 컬럼명 (`start_date`, `end_date`, `fact`, **`preds`**) 5개 도메인 동일 확인. 단 `pred`가 아니라 **`preds`** (s 붙음)
- [x] 도메인별 텍스트 sparsity 통계 — report no-info 22.8%~100%(Security 빈 파일), search no-info 0%~41.9% (도메인별 편차 큼)
- [x] 시간 범위 확인 — 위 표 참조
- [x] 다변량 변수 수 — Agriculture/Economy 3개, 나머지 1개 (OT만)
- [x] Time-MMD 업스트림 버그 자동 보정: Traffic report의 연도-경계 윈도우 3건에서 `end_date` 연도 +1y

### 텍스트-월 매칭 정책
텍스트 윈도우 `[start_date, end_date]`가 두 달에 걸친 경우 (Search 약 20%, Report 약 11%), **겹친 일수가 더 많은 달에 배정** (동률시 빠른 월). 윈도우가 한 달 안에 완전히 포함된 경우(80%+)는 어느 정책이나 동일.

### Memorization 처리 (후속)

CiK 방식 채택, 데이터 구축 후 적용:
- Setup-Clean: 원본 시계열
- Setup-Noise: Gaussian noise 표준편차 3% 추가

두 setup의 성능 차이로 memorization 영향 정량화.

자세한 내용은 `04_evaluation.md` 참조.
