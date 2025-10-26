# 릴리스 가이드

## 빠른 점검 목록
- `scripts/setup_env.sh` 또는 `scripts/dev/setup_env.sh`로 가상환경과 의존성을 재정비합니다.
- 스테이징 데이터로 `infopilot.py pipeline all`을 돌려 증분 파이프라인과 자동 학습이 정상 동작하는지 확인하고, 필요한 경우 `infopilot.py run scan/train/chat/watch`를 각각 재실행합니다.
- `python scripts/util/release_prepare.py --print`으로 KPI JSON과 메트릭 스냅샷이 정상 생성되는지 확인합니다.
- 회의/사진 에이전트 파이프라인을 각각 한 번씩 돌려 `summary.json`, `photo_report.json` 등 산출물을 검증합니다.
- 데스크톱 배포가 필요하면 `scripts/build_desktop_ui.ps1`(Windows) 또는 `python scripts/launch_desktop.py --bundle`과 같은 PyInstaller 플로우로 새 실행 파일을 만들고, 버전 태그와 함께 업로드합니다.

## 릴리스 노트 템플릿
```
- 핵심 개선 사항
  - 지식·검색 비서: …
  - 회의 비서: …
  - 사진 비서: …
  - 기타 운영/툴링: …
- 품질 & 테스트
  - 실행한 회귀/스모크 테스트, KPI 스냅샷 요약
- 배포 메모
  - 인덱스 재생성 여부, 정책 변경 사항, 알려진 제한 사항
```

필요한 항목만 채워 메일이나 게시글에 공유하고, 태그 시점의 실행 로그(예: `python scripts/util/release_prepare.py --print`)를 함께 저장하면 추적이 쉽습니다.
