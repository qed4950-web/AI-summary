# 사진 비서 실행 방법 (Runbook)

사진 비서는 폴더 경로 리스트를 입력 받아 태깅/중복 정리를 수행합니다. UI에서는 자동으로 트리거되지만 CLI에서도 확인할 수 있습니다.

## 1. 입력 준비

- 대상 폴더는 로컬 경로여야 하며 이미지 파일(`.jpg`, `.jpeg`, `.png`, `.heic`)을 포함해야 합니다.
- 출력 디렉터리를 지정하지 않으면 `data/photo_outputs/`(또는 `PHOTO_OUTPUT_DIR` 환경 변수)가 사용됩니다.

## 2. 환경 변수 (선택)

```bash
export PHOTO_OUTPUT_DIR="$PWD/data/photo_outputs"
```

필요 시 정책 태그를 `policy_tag` 컨텍스트로 전달할 수 있습니다.

## 3. CLI 실행 예시

`infopilot.py chat` 모드에서 사진 비서를 호출하려면 follow-up 프롬프트에 경로를 입력하면 됩니다.  
단독으로 실행하고 싶다면 아래 스니펫을 사용하세요.

```bash
python - <<'PY'
from core.agents import AgentRequest
from core.agents.photo import PhotoAgent, PhotoAgentConfig

agent = PhotoAgent(PhotoAgentConfig())
agent.prepare()
response = agent.run(
    AgentRequest(
        query="사진 정리해줘",
        context={
            "roots": ["~/Pictures/2025/01", "~/Pictures/2025/02"],
            "output_dir": "data/photo_outputs/january_review",
        },
    )
)
print(response.content)
print(response.metadata.get("report_path"))
PY
```

이 코드는 루트 두 곳을 대상으로 파이프라인을 돌리고, 결과 요약과 리포트 경로를 출력합니다.

## 4. 산출물 확인

```bash
cat data/photo_outputs/january_review/photo_report.json | jq .
```

JSON에는 베스트샷 경로, 중복 그룹, 정책 태그가 포함됩니다.

## 5. 트러블슈팅

- **폴더 없음**: 경로가 잘못되면 경고 로그가 찍히고 해당 폴더는 건너뜁니다. 루트 배열에 절대 경로를 지정하세요.
- **취소 처리**: UI에서 작업을 취소하면 `TaskCancelled`가 발생해 친절한 메시지를 반환합니다.
- **확장 요구**: 태깅/중복 알고리즘을 교체하고 싶으면 `PhotoPipeline._tag`, `_deduplicate` 를 커스터마이즈하세요.

> 대화형 모드에서는 `/photo` 명령어로 사진 비서를 호출한 뒤 안내에 따라 사진 폴더를 선택합니다.
