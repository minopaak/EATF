# 03. ROI 라벨링 파이프라인

## 왜 ROI인가

기존 평가가 전체 forecast horizon의 평균 MSE만 봄. 시계열의 95%가 평상시고 5%가 이벤트 구간이라면, 이벤트 구간에서 모델이 망가져도 평균에 묻혀 안 보임.

ROI를 분리해서 평가하면:
- 평상시 패턴 학습 능력 (Non-ROI MSE)
- 이벤트 처리 능력 (ROI MSE)

이 두 능력을 분리 측정 가능 → cross-domain에서 어느 능력이 더 transfer 안 되는지 입증.

## 라벨링 범위

**전체 데이터** (train + val + test) 모두 라벨링.

Track A에서 사전 라벨을 train에도 줘야 모델이 라벨 활용 학습 가능.

## 라벨 스키마

```json
{
  "domain": "Agriculture",
  "timestamp": "2020-03-01",
  "text_source": "report",
  "is_event": true,
  "event_summary": "USDA announces emergency drought relief",
  "expected_direction": "down",
  "expected_duration": "medium",
  "roi_start": "2020-03-01",
  "roi_end": "2020-09-01",
  "annotation_source": "pred_based",
  "iou_with_cpd": 0.72,
  "confidence": 0.85,
  "flags": []
}
```

## ROI 추정 방식: 계층적

```
For each fact 텍스트:
  Step 1: LLM이 is_event 판단
  
  If is_event:
    Step 2: pred 텍스트에 명시적 시간 정보 있는지 확인
    
    If pred에 시간 정보 있음 (예: "next 6 months"):
      → duration 추출
      → roi_end = roi_start + duration
      → annotation_source = "pred_based"
    
    Else (pred 모호 또는 없음):
      → LLM이 ROI 추정
      → CPD로 시계열 변동 검증
      → IoU 계산
      
      If IoU >= 0.5:
        → annotation_source = "llm+cpd_validated"
      Else:
        → annotation_source = "human_review_required"
```

### 왜 pred 우선?
Time-MMD의 pred는 인간 도메인 전문가가 작성한 거라 ROI 길이가 합리적. LLM의 generic guess보다 신뢰도 높음. 시간 범위 정보만 활용하면 causal leakage 없음.

### 왜 LLM + CPD 하이브리드?
- LLM only: 시계열 동역학과 무관할 수 있음
- CPD only: 텍스트 정보 무시
- 두 신호 cross-check로 노이즈 라벨 필터링

## ROI 시작/종료 시점

### 시작 시점
텍스트 end_date를 monthly로 변환한 시점. 단순하고 monthly 단위라 차이 없음.

### 종료 시점
같은 우선순위 (pred → LLM+CPD).

## 검증 기준: IoU 0.5

```
IoU = (LLM ROI ∩ CPD ROI) / (LLM ROI ∪ CPD ROI)
```

- IoU ≥ 0.5: auto-validated
- IoU < 0.5: human review

컴퓨터 비전 표준이고 시계열 event detection 연구들도 0.5 사용.

## Edge Case 처리

1. **인풋에 이벤트가 있을 때만 ROI 발동**: 이벤트가 input window 안에 있어야 valid. Forecast horizon에서 발생한 이벤트는 평가 제외.
2. **짧은 ROI (1개월)도 평가**
3. **ROI 길이 cap 없음**: Forecast horizon 안에 들어오는 부분만 자동 평가
4. **겹친 ROI는 union mask로 평가**: 어느 이벤트 영향인지 분리 안 함
5. **시계열 변동 안 보이는 이벤트도 valid**: `low_signal: true` flag만 표시

## 평가 셋업 (라인 1: Time-MMD 표준)

- Look-back window의 텍스트만 사용 (historical text)
- 이벤트가 input window 안에 발생 가능
- 모델이 이벤트 정보 + 시계열 초기 반응을 input으로 받음
- Forecast는 이벤트 영향의 진행 + 회복 예측

Future event description 셋업 (CiK, EventTSF 방식)은 future work.

## 구현 단계

### 1단계: LLM 라벨링
- 모델: GPT-4o (또는 오픈 모델)
- 입력: fact + pred 텍스트, recent time series values
- 출력: 위 스키마 JSON
- 도메인당 비용 추정 필요

### 2단계: CPD 검증
- 라이브러리: `ruptures`
- 알고리즘: PELT
- 도메인별 penalty 튜닝 필요

### 3단계: 사람 검증
- 도메인당 random sample 200개
- 3명 annotator
- Fleiss' κ 목표 0.7

### 4단계: 가이드라인 문서화
재현성 확보 위해 라벨링 criteria 명문화.

## 미정 사항

- LLM 프롬프트 디자인 구체화
- CPD penalty 도메인별 값
- Annotator 섭외 및 운영
- 라벨링 비용 산정
