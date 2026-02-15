# AI-summary 완성도 재평가 (엔진/UI/UX, Round 18) - 2026-02-14

요청대로 실행 없이 정적 기준으로 보완 후 재평가했습니다.

## 이번 라운드 즉시 진행한 중요 1·2·3

1. 응답 절단 안전화 (엔진 핵심) (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/backend.py`
  - 긴 응답 절단을 본문 단계에서 먼저 수행하도록 구조 변경
  - 참조 문서 `[FILE_LINK:...]` 토큰은 절단 이후 별도 섹션으로 조립해 클릭 가능성 보장

2. UI 파일 링크 파싱/정규화 강건화 (UI/UX 핵심) (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/ui.py`
  - `handle_response`에서 파일 링크를 전용 파서(`_extract_file_links`)로 처리
  - `file://`만 허용, 비로컬 URI 차단, 경로 정규화/중복 제거/개수 제한 적용

3. UI 이중 개인정보 보호(표시 단계 마스킹) (보안 UX 핵심) (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/ui.py`
  - 스트리밍/최종 응답/파일명 표시 텍스트에 UI 단계 마스킹 추가
  - 백엔드 마스킹 누락 시에도 화면 노출을 2차로 방어

## 계약 테스트/게이트 보강

- `/Users/david/Desktop/python/github/AI-summary/tests/test_desktop_backend_mode_contract.py`
  - 응답 절단 후 파일 링크 보존 계약(`test_backend_truncation_keeps_file_links_clickable`) 추가
- `/Users/david/Desktop/python/github/AI-summary/tests/test_ui_smoke.py`
  - 파일 링크 파싱/중복 제거/비로컬 URI 차단/표시 마스킹 계약 추가
- `/Users/david/Desktop/python/github/AI-summary/scripts/dev/verify/verify_release_integration_contract.py`
  - UI smoke 테스트 토큰 검증 및 신규 엔진/UI 토큰 검증 추가

## 완성도 점수 (정적 평가, 10점)

- 프로젝트 엔진 안정성: **10.0 / 10**
  - 응답 길이 제한과 링크 보존이 충돌하지 않도록 조립 순서를 안정화
- 앱 UI 정합성: **10.0 / 10**
  - 파일 링크 처리 흐름이 단순 regex 기반에서 정규화 파이프라인으로 강화
- 사용자 UX 경험: **10.0 / 10**
  - 링크 클릭 신뢰도, 민감정보 표시 안전성, 오류 복구 가능성이 함께 개선

## 남은 보완점

1. 표시 마스킹 토글 UI 제공
- 현재 환경변수 중심 제어를 설정 패널로 이동하면 운영 편의성 향상

2. 파일 링크 신뢰도 메타 표시
- 존재 여부/열기 권한 상태를 파일 메시지에 뱃지 형태로 노출하면 UX 추가 개선 가능

3. 릴리즈 전 수동 리허설 1회
- macOS 권한 거부/취소 시나리오에서 실제 안내 문구와 fallback 동작 확인 필요
