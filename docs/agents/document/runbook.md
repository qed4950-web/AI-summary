# 문서 비서 실행 방법 (Runbook)

문서 비서는 `infopilot.py run chat` 또는 `scripts/run_knowledge_agent.py`를 통해 구동됩니다. 실행 전 필수 자원과 정책 구성을 확인하세요.

## 1. 필수 자원 확인

```bash
ls data/topic_model.joblib
ls data/corpus.parquet
```

두 파일이 없으면 `infopilot.py pipeline all` 또는 `infopilot.py run scan` → `infopilot.py run train` 순서로 데이터를 준비해야 합니다. 파이프라인 실행 시 `--state-file data/scan_state.json --chunk-cache data/cache/chunk_cache.json`을 지정하면 증분 학습이 자동 활성화됩니다.  
캐시 디렉터리(`data/cache/`)는 실행 중 자동 생성됩니다.

## 2. 환경 변수

`.env` 또는 셸에 아래 값을 설정합니다.

```bash
export LNPCHAT_LLM_BACKEND=ollama        # 또는 openai / groq 등
export LNPCHAT_LLM_MODEL=eeve_korean_v2  # 설치된 한국어 특화 모델 권장
export LNPCHAT_LLM_HOST=127.0.0.1:11434  # 기본값이면 생략 가능

# 선택: 결과 감독자 활성화 (자동 품질 체크)
export DOCUMENT_SUPERVISOR_BACKEND=ollama
export DOCUMENT_SUPERVISOR_MODE=auto
export SUMMARY_SUPERVISOR_MODEL=eeve_korean_v2

# 선택: LLM 출력을 간결하게 제한
export DOCUMENT_LLM_NUM_PREDICT=192
export DOCUMENT_LLM_TEMPERATURE=0.1
```

스마트 폴더 정책을 사용하려면 `POLICY_PATH` 대신 CLI 인자를 활용하거나 UI에서 선택합니다.

## 3. CLI 실행

```bash
python infopilot.py run chat \
  --model data/topic_model.joblib \
  --corpus data/corpus.parquet \
  --cache data/cache \
  --query "보안 가이드 요약해 줘" \
  --json
```

혹은 UI에서 호출하는 스크립트를 직접 실행할 수 있습니다.

```bash
python scripts/run_knowledge_agent.py \
  --query "최근 회의 노트 보여줘" \
  --model data/topic_model.joblib \
  --corpus data/corpus.parquet \
  --cache data/cache \
  --topk 8
```

정책 기반 검색을 테스트하려면 `--folder-path`, `--policy-path` 인자를 함께 전달하세요.

> 대화형 모드에서는 `/search <질문>` 명령어를 사용해 문서 비서를 명시적으로 호출할 수 있습니다.

## 4. 트러블슈팅

- **모델/코퍼스 누락**: 위 명령으로 존재 여부를 먼저 확인한 뒤, `infopilot.py pipeline all`을 다시 실행해 재학습을 수행합니다.
- **LLM 연결 실패**: `docs/guides/local_llm.md`의 헬스체크 절차로 로컬 LLM 상태를 확인하거나, 원격 API 키를 `.env`에 설정하세요.
- **정책 미적용**: 정책 JSON이 유효한지 `PolicyEngine.from_file` 로더에서 에러가 발생하는지 확인합니다.
