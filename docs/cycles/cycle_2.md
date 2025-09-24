## Cycle 2 – P1 (지식·검색 고도화)

### 목표
- 문서 임베딩/요약 파이프라인을 일원화하고, 검색 결과에 근거 하이라이트를 추가
- 전역/폴더 스코프 전환 UX와 작업 센터 MVP를 설계/구현
- 정책 기반 검색/응답 흐름을 LLM 서비스와 프론트엔드까지 확장

- [x] 임베딩·요약 결과 프리뷰 하이라이트 개선 및 정책 기반 필터링 도입
- [x] 전역/폴더 스코프 전환 CLI 옵션(`--scope`) 추가 (auto/policy/global)
- [ ] 작업 센터 초안 (최근 요약/액션/알림) 구현
- [ ] LLM/프론트엔드에서 정책 준수 및 컨텍스트 표시

### 산출물
- 스마트 폴더 정책 엔진 확장: `infopilot_core/data_pipeline/policies/engine.py`
- CLI 정책/스코프 연동: `infopilot.py`
- 정책 적용 대화 세션: `infopilot_core/conversation/lnp_chat.py`

### 다음 단계 체크리스트
- Cycle 2 완료 후 사용자 피드백 세션 및 Cycle 3(회의 비서) 준비
- 문서화: `docs/cycles/cycle_2.md`에 주요 결정/이슈 기록
