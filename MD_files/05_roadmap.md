# 05. 로드맵 및 미결정 사항

## Phase별 진행 계획

### Phase 1: 데이터 통합 (1-2주)
- [x] 설계 문서 작성
- [x] `build_dataset.py` 작성 (도메인 일반화, 텍스트 majority-overlap 매칭, Time-MMD 버그 보정)
- [x] 5개 도메인 확장 (Climate weekly 제외; Agriculture/Economy/Security/SocialGood/Traffic)
- [x] 도메인별 통계 (텍스트 sparsity, 시간 범위, 변수 수) — `data/processed/` 저장 시 로그 출력
- [ ] EDA 노트북 (시계열 plot, 텍스트 length 분포)

### Phase 2: 파일럿 스터디 (1-2주)
**목표**: 파이프라인 sanity check + in-domain 베이스라인 측정. **헤드라인 cross-domain 검증은 아님** (그건 Phase 5 — multimodal + 텍스트 transfer가 본 게임). TS-only LODO는 reference로만.

- [x] PatchTST, DLinear 베이스라인 (vendored) 셋업
- [x] In-domain 베이스라인 측정 (5 도메인 × {DLinear, PatchTST}, L=8 H={6,8,10,12}, 3-seed)
- [x] **파이프라인 검증**: MM-TSFlib 데이터로 우리 코드 돌려 논문 재현(Traffic·SocialGood 정확 일치) — 우리 파이프라인 정합성 입증
- [ ] (선택) TS-only LODO reference 테이블 (L=8 + 트림 데이터로 5×5 재실행) — 헤드라인은 아니지만 사례 비교용으로 한 번 정리

### Phase 3: 라벨링 파이프라인 (3-4주)
- [ ] LLM 프롬프트 디자인 + 반복 개선
- [ ] CPD 도메인별 파라미터 튜닝
- [ ] 라벨링 가이드라인 작성
- [ ] Annotator 섭외 (3명)
- [ ] Calibration sample 50개로 트레이닝

### Phase 4: 전체 라벨링 (4-6주)
- [ ] 6개 도메인 LLM 자동 라벨링
- [ ] CPD 검증
- [ ] 도메인당 200 sample human IAA
- [ ] Adjudication
- [ ] κ 측정 및 보고

### Phase 5: 베이스라인 평가 (4-5주)
- [ ] 6개 모델 dataloader 수정
- [ ] In-domain sanity check (원본 보고치 대비 ±5%)
- [ ] 3 protocol × 2 track 전체 실행
- [ ] VoT 변종 ablation

### Phase 6: 분석 및 작성 (2주)
- [ ] 4가지 핵심 발견 검증
- [ ] 도메인 거리 분석
- [ ] Memorization setup 비교
- [ ] 논문 초안

## 미결정 사항

### 라벨링
- LLM 프롬프트 구체적 디자인
- LLM 모델 선택 (GPT-4o vs 오픈 모델)
- 라벨링 비용 산정
- Annotator 섭외 방식 (학생, 크라우드소싱 등)
- IAA 측정 방법론 (Fleiss' κ vs Cohen's κ)

### 평가
- Track A 사전 라벨의 모델별 입력 형식 표준화
- 도메인 거리 측정 방법 (시계열 통계 vs 텍스트 임베딩)
- VoT HIC 변종 평가 시 KB 운영 방식

### 자원
- GPU 자원 확보
- API 비용 예산
- Annotator 인건비

### 데이터
- CPD penalty 도메인별 튜닝 방식
- 6개 도메인의 컬럼명 일관성 (사전 검증 필요)
- Memorization 적용 시점 (베이스라인 평가 직전)

## 일정 요약

| Phase | 기간 | 주요 산출물 |
|-------|------|------------|
| 1 | 1-2주 | 통합 CSV (6개 도메인) |
| 2 | 1-2주 | 파일럿 결과, go/no-go 결정 |
| 3 | 3-4주 | 라벨링 파이프라인 |
| 4 | 4-6주 | 전체 라벨링 + IAA |
| 5 | 4-5주 | 베이스라인 결과 |
| 6 | 2주 | 논문 초안 |

**총 약 4개월**

## 위험 요소

| 위험 | 영향 | 대응 |
|------|------|------|
| Cross-domain gap이 안 보임 | 데이터셋 motivation 무너짐 | Phase 2 파일럿으로 조기 검증 |
| IAA κ < 0.7 | 라벨 quality 신뢰도 | 가이드라인 보강, 라벨링 기준 단순화 |
| Memorization 영향 큼 | 평가 신뢰도 | CiK 방식 + cutoff 분석 |
| 베이스라인 호환 안 됨 | 평가 불가 | 모델당 1-2일씩 확보, 어려우면 제외 |
| Fidel-TS 비판 | Reviewer 우려 | 차별점 명확히, 보완 관계 포지셔닝 |

## 다음 액션 (업데이트: 2026-05)

Phase 1·2 사실상 완료 (텍스트 시대 trim + in-domain 베이스라인 + 파이프라인 검증). **다음은 Phase 3 (ROI 라벨링 파이프라인)** — 헤드라인 cross-domain 실험(텍스트/이벤트 transfer)은 multimodal 모델이 필요하고 그건 ROI 라벨을 전제로 함. 소소한 잔여: EDA 노트북, TS-only LODO reference 테이블(원하면).

**왜 순서가 바뀌었나**: 원래 Phase 2 다음 Phase 3였는데, "Phase 2 cross-domain gap 검증"의 원래 의도(TS-swap LODO)가 우리 contribution과 정렬되지 않음을 확인 (2026-05). 단순 TS 도메인 간 transfer는 동역학이 달라 trivial fail이라 인사이트 없고, 우리 핵심 가설("case-based 멀티모달은 source 이벤트 memorize, transfer 실패")은 텍스트/이벤트 모달리티의 transferability 질문이라 multimodal 모델에서만 측정 가능. 따라서 헤드라인 검증은 Phase 5로 미뤄지고, 그 전제인 Phase 3가 다음.
