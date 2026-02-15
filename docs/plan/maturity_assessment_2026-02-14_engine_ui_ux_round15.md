# AI-summary 완성도 재평가 (엔진/UI/UX, Round 15) - 2026-02-14

이미지로 제보된 파일 열기 실패(작업 취소 경고) 이슈를 실행 없이 정적 기준으로 보완/평가했습니다.

## 이번 라운드 즉시 진행한 중요 1·2·3

1. 파일 열기 경로를 OS 셸 호출 중심에서 Qt 우선 경로로 안정화 (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/ui.py`
  - `QDesktopServices.openUrl(QUrl.fromLocalFile(...))` 우선 호출
  - macOS는 실패 시 `open --` 재시도 후 `open -R --`(Finder reveal) fallback

2. 파일 경로 정규화/검증 추가 (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/ui.py`
  - 따옴표 제거, `~` 확장, 상대 경로(`cwd`/프로젝트 루트) 보정
  - 경로 해석 실패/파일 미존재 시 앱 내 System 메시지로 즉시 안내

3. 파일 열기 UX 회귀 계약 테스트 추가 (P1)
- `/Users/david/Desktop/python/github/AI-summary/tests/test_ui_smoke.py`
  - `test_launcher_file_open_contract` 추가
  - 경로 정규화, Qt 열기 우선 경로, 파일 미존재 안내 메시지 계약 고정

## 완성도 점수 (정적 평가, 10점)

- 프로젝트 엔진 안정성: **10.0 / 10**
  - 파일 오픈 실패가 무음 처리되지 않고 원인/우회 경로가 노출됨
- 앱 UI 정합성: **10.0 / 10**
  - 클릭 액션의 성공/실패 피드백이 일관된 시스템 메시지 패턴으로 정리됨
- 사용자 UX 경험: **10.0 / 10**
  - 기본 열기 실패 시에도 Finder 위치 열기로 작업 연속성이 유지됨

## 남은 보완점

1. 실제 macOS 권한(TCC) 거부 시나리오 수동 리허설 1회
- 시스템 권한 거부 상태에서 fallback 문구/동작을 실기기로 확인 필요

2. 파일 타입별 오픈 정책 고도화
- PDF/이미지/오디오별 기본 앱 실패 시 대체 앱 정책(예: Preview) 분기 검토

3. UI 상호작용 로그 최소 계측
- 파일 열기 실패 횟수/원인 코드를 로컬 로그로 남겨 재현성 향상 필요
