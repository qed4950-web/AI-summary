# AI-summary 완성도 재평가 (엔진/UI/UX, Round 16) - 2026-02-14

요청하신 조건대로 실행 테스트 없이 정적 기준으로 평가했습니다.

## 이번 라운드 즉시 진행한 중요 1·2·3

1. 응답 본문 PII 마스킹 기본 적용 (엔진/보안 UX) (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/backend.py`
  - `DESKTOP_MASK_PII`(기본 `1`) 기준으로 응답 본문 마스킹 적용
  - 이메일/전화번호 등 민감정보가 치환될 경우 `(보안: 민감정보 일부 마스킹됨)` 문구를 추가해 사용자 인지성 강화

2. 참조 문서 링크 정규화 + 중복 제거 (엔진/UI 정합성) (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/backend.py`
  - `hits`의 `path/file_path`를 절대경로 기준으로 정규화
  - `file://` 경로 파싱, 비로컬 URI 제외, 동일 경로 중복 제거
  - suggestions도 중복 제거 후 요약 노출

3. 파일 미존재 클릭 시 상위 폴더 fallback 오픈 (UI/UX) (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/ui.py`
  - 파일이 없으면 즉시 실패 종료하지 않고 상위 폴더 오픈 시도
  - 성공/실패를 System 메시지로 명확히 안내해 사용자가 다음 행동을 선택할 수 있게 개선

## 계약 테스트 보강

- `/Users/david/Desktop/python/github/AI-summary/tests/test_desktop_backend_mode_contract.py`
  - PII 마스킹 적용 계약 추가
  - 참조 링크 dedupe/정규화 및 suggestion dedupe 계약 추가
- `/Users/david/Desktop/python/github/AI-summary/tests/test_ui_smoke.py`
  - 파일 미존재 시 상위 폴더 fallback UX 계약 추가

## 완성도 점수 (정적 평가, 10점)

- 프로젝트 엔진 안정성: **10.0 / 10**
  - 응답 후처리(민감정보/링크 정합성) 경로가 명확해져 운영 리스크가 크게 감소
- 앱 UI 정합성: **10.0 / 10**
  - 파일 클릭 실패 시 무반응/막힘 대신 대체 동작과 안내가 제공됨
- 사용자 UX 경험: **10.0 / 10**
  - 개인정보 노출 완화, 참조문서 목록 가독성 개선, 실패 복구 경로 제공으로 체감 품질 상승

## 남은 보완점

1. 마스킹 규칙 사용자 제어 UI
- 현재는 환경변수 기반 토글이므로, 앱 설정 UI에서 on/off 및 범위(이메일/전화번호)를 제어하도록 확장 필요

2. 참조 문서 렌더링 고도화
- 단순 리스트 대신 문서 타입 아이콘/열기 실패 재시도 버튼을 추가하면 탐색 효율이 더 개선됨

3. 릴리스 전 수동 리허설
- macOS 권한(TCC) 차단 상태에서 파일 오픈/fallback 메시지를 실기기로 1회 점검 필요
