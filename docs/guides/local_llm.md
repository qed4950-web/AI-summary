# Local LLM Setup Guide

이 문서는 LNPChat에서 경량 LLM(예: Ollama)을 활용하기 위한 환경 설정 방법을 정리합니다.

## 1. Ollama 설치
- macOS: `brew install ollama`
- Linux: https://ollama.com/download 의 안내를 따릅니다.

설치 후 `ollama --version` 으로 동작을 확인하세요.

## 2. 모델 다운로드
기본 예제 모델은 `llama3` 입니다. 필요 시 다음 명령으로 내려받습니다.
```bash
ollama pull llama3
```
다른 모델을 사용할 경우 LNPChat 환경 변수에서 모델 이름을 수정하세요.

## 3. 환경 변수 설정
`LNPCHAT_LLM_BACKEND`, `LNPCHAT_LLM_MODEL`, `LNPCHAT_LLM_HOST` 환경 변수를 설정하면 LNPChat이 자동으로 로컬 LLM을 사용합니다.

예시 (`.env` 또는 셸 환경):
```bash
export LNPCHAT_LLM_BACKEND=ollama
export LNPCHAT_LLM_MODEL=llama3
export LNPCHAT_LLM_HOST=127.0.0.1:11434  # 기본값이면 생략 가능
```

## 4. 동작 확인
```bash
python scripts/check_local_llm.py ollama --model llama3
```

정상적으로 설치되어 있다면 "ollama ready" 메시지가 출력됩니다.

## 5. LNPChat에서 사용
이후 `infopilot.py chat` 또는 데스크톱 UI를 통해 LNPChat을 사용할 때 자동으로 로컬 LLM 요약이 활성화됩니다. 백엔드 미동작 시에는 기존 검색 리스트로 폴백합니다.

## 6. 주의 사항
- 로컬 LLM 추론은 시스템 자원을 많이 사용할 수 있으므로 필요에 따라 모델 크기를 조정하세요.
- 배경 서비스가 필요한 경우 `ollama serve` 를 실행해 서버를 띄우고, `LNPCHAT_LLM_HOST` 를 해당 주소로 지정하세요.
