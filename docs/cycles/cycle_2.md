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
- 스마트 폴더 정책 엔진 확장: `core/data_pipeline/policies/engine.py`
- CLI 정책/스코프 연동: `infopilot.py`
- 정책 적용 대화 세션: `core/conversation/lnp_chat.py`

#### 2025-10-17 점검
- 작업 센터 대시보드는 아직 구현 전(`ui/screens/home_screen.py`), `docs/ux/improvements.md`의 요구 사항을 기반으로 MVP 설계 필요.
- 정책 컨텍스트/스코프 노출은 CLI(`core/conversation/lnp_chat.py`)에서만 제공되므로, UI(chat 화면)에 동일 기능을 이식하는 작업이 남아 있음.

### 다음 단계 체크리스트
- Work Center MVP 화면(`ui/screens/work_center.py` 가칭) 설계 및 정책 스코프/컨텍스트 노출 구현
- LLM/프론트엔드 정책 준수 테스트와 QA 시나리오 작성
- Cycle 2 완료 후 사용자 피드백 세션 및 Cycle 3(회의 비서) 준비
- 문서화: `docs/cycles/cycle_2.md`에 주요 결정/이슈 기록
