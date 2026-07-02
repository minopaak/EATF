# 05. 로드맵

메커니즘-증강 예측 파이프라인 기준. (옛 ROI 라벨링·cross-domain 로드맵은 폐기)

## Phase별 진행

### Phase 1: 예측 데이터 구축 — ✅ 완료
- [x] `build_dataset.py` (도메인 일반화, 텍스트 majority-overlap 매칭, Time-MMD 버그 보정)
- [x] 5개 도메인 CSV (`data/processed/`, 텍스트 시대 trim)
- [x] in-domain 베이스라인 측정 (7 백본 × {unimodal, MM-TSFlib}, L=8, H={6,8,10,12}, 3-seed) — 이제 **local branch 참조선**
- [x] 파이프라인 정합성 검증 (MM-TSFlib 데이터로 논문 재현)

### Phase 2: Mechanism Pool — 🔄 진행 중 (별도 repo)
- pool 생성은 `knowledge_pool` 담당 (본 프로젝트 범위 밖).
- [x] pilot: Economy DK 140 + HE 18 entry
- [ ] 나머지 4개 도메인 DK/HE 구축 (knowledge_pool)
- [ ] EATF 소비 브리지: pool JSON 로드 유틸

### Phase 3: Retrieval 인프라 — ⬜ 다음
- [ ] pool `content` 임베딩 + 검색 인덱스 구축
- [ ] top-K 검색 (HE cutoff 필터 `max(evidence.year) < test_time`)
- [ ] query = 예측 맥락 기반 (query 생성 방식은 초기엔 단순 고정, 정교화는 Phase 5로 보류)

### Phase 4: Mechanism-aware Forecasting 모델
- [ ] **Local branch**: 기존 TS 백본 + 정렬 텍스트 융합 재사용 (MM-TSFlib 융합 개조)
- [ ] **Global branch**: retrieval → (reasoning) → mechanism encoder → Zᴹ
- [ ] **Fusion**: gated cross-attention (Zᴸ, Zᴹ) → head → 예측

### Phase 5: 학습 (Inner/Outer loop)
- [ ] Inner: neural 모듈을 forecasting loss로 학습 (query 고정)
- [ ] Outer: val high-error 케이스로 query-writing 규칙 개선 (forecast-aware adaptation)

### Phase 6: 평가·분석·작성
- [ ] baseline 비교 (numerical-only / aligned-text / ours)
- [ ] ablation (특히 event vs mechanism retrieval)
- [ ] memorization (CiK clean/noise)
- [ ] 논문 초안

## 위험 요소

| 위험 | 영향 | 대응 |
|------|------|------|
| 메커니즘 검색이 예측을 안 도움 | 핵심 주장 무너짐 | 이른 시점에 Phase 3+간단 융합으로 조기 신호 확인 |
| pool coverage/품질 부족 | 검색 무의미 | knowledge_pool에서 도메인별 saturate 확인 |
| HE 시점 누수 | 평가 신뢰도 | cutoff 필터 엄격 적용·검증 |
| pool 텍스트 memorization | 평가 신뢰도 | CiK noise setup 비교 |

## 미결정

- query 생성 방식 (초기 고정 프롬프트 → forecast-aware 적응까지 얼마나)
- reasoning agent 사용 여부·강도 (raw text vs reasoning ablation으로 결정)
- mechanism encoder / fusion 구체 구조
- retrieval 임베딩 모델 (frozen LLM vs 전용 retriever)
