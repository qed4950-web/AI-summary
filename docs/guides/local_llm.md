# 로컬 LLM 연동 가이드

LNP Chat이 Ollama 기반 LLM을 사용하도록 설정하는 절차를 요약했습니다. 아래 단계는 macOS·Linux 기준이며, Windows 프리뷰 빌드는 Ollama 공식 문서를 참고하세요.

## 1. Ollama 설치 & 확인

```bash
# macOS (Homebrew)
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

ollama --version
```

버전이 출력되면 설치가 완료된 것입니다.

## 2. 모델 다운로드

```bash
ollama pull llama3:8b-instruct-q4   # 권장: 8B Instruct 양자화 모델
ollama pull llama3.1:8b-instruct-q4 # 필요한 경우 추가 모델
ollama list
```

`ollama list`를 통해 설치된 모델을 확인하세요. `llama3:8b-instruct-q4`를 설치하면 기본 별칭 `llama3`로 실행할 수 있습니다 (`ollama run llama3`).

## 3. 환경 변수 구성

CLI 또는 `.env` 파일에서 다음 변수를 지정하면 LNP Chat이 자동으로 로컬 LLM을 사용합니다.

```bash
export LNPCHAT_LLM_BACKEND=ollama
export LNPCHAT_LLM_MODEL=llama3          # 설치한 모델 이름 (예: llama3)
export LNPCHAT_LLM_HOST=127.0.0.1:11434  # 기본값이면 생략 가능
```

## 4. 연결 테스트

Ollama 데몬이 실행 중인지 확인한 뒤 헬스체크 스크립트를 실행합니다.

```bash
ollama serve &               # 백그라운드 실행 (이미 실행 중이면 생략)
python scripts/check_local_llm.py --backend ollama --model llama3
# 다른 모델명을 테스트하고 싶다면 --model llama3.1:8b-instruct-q4 처럼 지정
```

모두 통과하면 LNP Chat이 LLM을 사용할 준비가 된 것입니다.

## 5. LNP Chat에서 사용

- **데스크톱 앱**  
  앱 실행 후 **대화 비서 → ⚙️ 설정**에서 `LLM 백엔드=ollama`를 선택하면 설치된 모델 목록이 표시됩니다. 목록이 비어 있으면 `ollama list`로 모델을 확인하거나 `목록 갱신` 버튼을 눌러 주세요.

- **CLI**  
  환경 변수를 설정한 세션에서 다음 명령으로 대화 모드를 실행합니다.

  ```bash
  python infopilot.py chat \
    --model data/topic_model.joblib \
    --corpus data/corpus.parquet \
    --cache data/cache
  ```

  LLM 연결에 실패하면 안전 모드로 전환되어 검색 결과만 제공되므로 설정을 다시 점검하세요.

## 6. 트러블슈팅

- **연결 실패 / 타임아웃**  
  `ollama serve`가 실행 중인지, 방화벽에서 11434 포트가 허용되어 있는지 확인하세요. 원격 호스트를 사용할 경우 `LNPCHAT_LLM_HOST`에 해당 주소를 지정합니다.

- **리소스 부족**  
  모델 크기에 따라 CPU·RAM 사용량이 크게 증가할 수 있습니다. 필요 시 더 작은 모델을 선택하거나, Ollama의 `set parameter` 기능으로 메모리 사용량을 조정하세요.

- **목록이 비어 있음**  
  `ollama list` 출력이 정상이라면 데스크톱 앱에서 **목록 갱신** 버튼을 눌러 캐시를 지우고 다시 로드하세요. CLI에서는 환경 변수 이름이 정확한지 재확인합니다.

이제 LNP Chat이 로컬 LLM과 함께 자연어 답변을 생성할 수 있습니다.
