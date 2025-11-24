# AI-summary

AI-summary는 로컬 문서 컬렉션을 스캔·학습하고, 의미 기반 검색·대화·요약을 제공하는 데스크톱/CLI 툴킷입니다.

## 1. 개요

- **파이프라인**: `infopilot.py` 
로 스캔 → 학습 → 대화 모드를 일괄 실행  
- **검색/대화**: `core/search/retriever.py`, `core/conversation/lnp_chat.py`가 BGE-m3 SentenceTransformer 임베딩과 정책 필터링을 결합  
- **도메인 에이전트**: 회의(STT→요약), 사진(중복/태깅) 비서를 `core/agents/`에서 제공  
- **UI**: `ui/`의 CustomTkinter 앱으로 주요 기능을 한 화면에서 제어  
- **문서화**: 아키텍처, 가이드, 실험 기록은 `docs/` 하위 폴더에 정리

## 2. 리포지토리 구조

```
core/                  데이터 파이프라인·검색·에이전트 구현
  ├─ agents/           회의·사진 등 도메인 기능
  ├─ conversation/     LNP Chat 엔진
  ├─ data_pipeline/    스캔·정제·학습 파이프라인
  └─ search/           의미 검색기 & 인덱스
ui/                    CustomTkinter 데스크톱 앱
data/                  실행 중 생성되는 산출물 (Git 관리 제외)
models/                로컬 모델 캐시 (Git 관리 제외)
scripts/               CLI/빌드/유틸 스크립트
docs/                  agents / architecture / guides / plan / process / research / roadmap / ux
tests/                 pytest 기반 단위·통합 테스트
```

`data/`, `models/`는 `.gitignore`에 포함되어 있으므로 필요한 경우 `.gitkeep`으로 디렉터리만 유지합니다.

## 3. 빠른 시작

### 3.1 필수 요건

- Python 3.9 이상
- (권장) 가상환경 사용

### 3.2 환경 준비

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.3.0" "torchvision==0.18.0" "torchaudio==2.3.0"
```

### 3.3 환경 변수(.env)

```bash
cp .env.example .env   # scripts/setup_env.sh 실행 시 자동 생성되기도 합니다.
```

생성된 `.env`를 열어 LLM/회의 비서 등에 필요한 값을 조정하세요. 주요 항목은 아래와 같습니다.

- `LNPCHAT_LLM_BACKEND`, `LNPCHAT_LLM_MODEL`, `LNPCHAT_LLM_HOST`
- `MEETING_OUTPUT_DIR`, `MEETING_ANALYTICS_DIR`
- `MEETING_SUMMARY_*`, `MEETING_STT_*`, `MEETING_WAV2VEC2_*`, `MEETING_RAG_*` 등 회의 비서 옵션

필요한 값만 유지하고 나머지는 공란으로 두어도 됩니다.

### 3.4 파이프라인 실행

```bash
# 0) 전체 파이프라인 한 번에 (스캔→학습→필요 시 chat)
python infopilot.py pipeline all \
  --out data/found_files.csv \
  --corpus data/corpus.parquet \
  --model data/topic_model.joblib \
  --cache data/cache \
  --state-file data/scan_state.json \
  --chunk-cache data/cache/chunk_cache.json \
  --launch-chat

# 또는 개별 단계
python infopilot.py run scan --out data/found_files.csv
python infopilot.py run train \
  --scan_csv data/found_files.csv \
  --corpus data/corpus.parquet \
  --model data/topic_model.joblib \
  --state-file data/scan_state.json \
  --chunk-cache data/cache/chunk_cache.json \
  --async-embed --embedding-concurrency 2
python infopilot.py run chat \
  --model data/topic_model.joblib \
  --corpus data/corpus.parquet \
  --cache data/cache \
  --lexical-weight 0.35
python infopilot.py run watch \
  --cache data/cache \
  --corpus data/corpus.parquet \
  --model data/topic_model.joblib
```

`pipeline all`은 scan/train을 자동으로 호출하고 증분 상태(`data/scan_state.json`)와 문서 해시 캐시(`data/cache/chunk_cache.json`)까지 유지하므로, 반복 실행 시 변경된 문서만 재처리합니다. 개별 단계는 `run <command>` 그룹으로 사용할 수 있으며, 필요한 경우 `--state-file`, `--chunk-cache`, `--async-embed/--no-async-embed`, `--embedding-concurrency` 등의 옵션으로 증분·성능 전략을 조정할 수 있습니다.

보조 명령도 함께 제공합니다.

```
python infopilot.py logs show         # MLflow/psutil 로그 tail
python infopilot.py logs clean --drift --resource
python infopilot.py model quantize --model sentence-transformers/... --output models/sbert.onnx
python infopilot.py drift check --scan-csv data/found_files.csv --corpus data/corpus.parquet
python infopilot.py drift reembed --path /docs/2023/report.docx --scan-csv ... --corpus ...
```

> 대화 비서에서 회의나 사진 정리를 요청하면 자동으로 해당 전용 비서를 호출합니다. CLI는 최근에 사용한 경로 목록을 보여 주고, 번호 선택 또는 직접 입력으로 오디오/폴더를 지정할 수 있는 프롬프트를 제공합니다. 추가 정보가 필요한 경우 후속 질문이 이어집니다.

### 3.4.1 자주 쓰는 명령 모음 (하단 북마크)

```bash
# 1) 임베딩 스캔 + 코퍼스/모델 학습
python infopilot.py run scan   --out data/found_files.csv
python infopilot.py run train  --scan_csv data/found_files.csv --corpus data/corpus.parquet --model data/topic_model.joblib --state-file data/scan_state.json --chunk-cache data/cache/chunk_cache.json --async-embed --embedding-concurrency 2

# 2) 통합 파이프라인 한 번에 + 완료 후 CLI 켜기
python infopilot.py pipeline all --out data/found_files.csv --corpus data/corpus.parquet --model data/topic_model.joblib --cache data/cache --state-file data/scan_state.json --chunk-cache data/cache/chunk_cache.json --launch-chat

# 3) 로컬 대화/검색 에이전트 실행
python infopilot.py run chat   --model data/topic_model.joblib --corpus data/corpus.parquet --cache data/cache --lexical-weight 0.35
python infopilot.py run watch  --cache data/cache --corpus data/corpus.parquet --model data/topic_model.joblib  # 신규 파일 자동 스캔·증분 임베딩

# 4) 회의 요약/파이프라인 UI 실행
python ui/app.py           # 데스크톱 UI (PySide6)
python scripts/run_all.sh  # FastAPI 백엔드 + webapp(Vite) 동시 실행
```

### 3.5 Prefect DAG 실행

`scripts/prefect_dag.py`는 scan→train→index→(선택)평가 단계를 Prefect 2.x Flow로 래핑합니다. Prefect를 설치했다면 아래와 같이 단일 명령으로 실행하거나 Prefect UI/에이전트에 배포할 수 있습니다.

```bash
python scripts/prefect_dag.py \
  --root /Users/me/Documents \
  --scan-csv data/found_files.csv \
  --corpus data/corpus.parquet \
  --model data/topic_model.joblib \
  --cache data/cache \
  --evaluation-cases data/eval/cases.jsonl \
  --use-prefect
```

`--use-prefect`를 생략하면 동일한 Runner를 순수 Python 모드로 실행해 MLflow/psutil 세션과 독립적으로 사용할 수 있습니다. Prefect Deployment를 만들고 싶다면 `prefect deployment build scripts/prefect_dag.py:prefect_pipeline_flow --name ai-summary` 같은 표준 Prefect 명령을 재사용하세요.

### 3.6 FastAPI 파이프라인 서버

자동화된 스케줄링이나 원격 제어가 필요하면 `scripts/api_server.py`로 FastAPI 서버를 띄울 수 있습니다.

```bash
python scripts/api_server.py
# POST http://127.0.0.1:8080/pipeline/run  {"scan_csv":"data/found_files.csv", ...}
# GET  http://127.0.0.1:8080/pipeline/status
# POST http://127.0.0.1:8080/pipeline/cancel
```

서버는 내부적으로 `scripts/prefect_dag.py`에서 제공하는 Runner를 재사용하며, 단계별 진행 상황/결과를 JSON으로 제공합니다. UI의 학습 화면에서도 동일한 API를 호출할 수 있도록 새로운 "FastAPI 제어" 패널을 추가했습니다.

## 4. 데스크톱 앱

```bash
python scripts/launch_desktop.py          # 개발 중에는 python ui/app.py
```

앱에서는 다음 화면을 제공합니다.

- 홈 대시보드: 모델/코퍼스 상태와 바로가기
- 대화 비서: 문서 기반 Q&A + LLM 요약(예: Ollama)
- 지식·검색 비서: 의미 검색 + 필터링
- 전체 학습 & 증분 업데이트: CLI 파이프라인을 GUI로 래핑
- FastAPI 제어 패널: UI에서 바로 파이프라인 API 실행/상태 확인/취소
- 회의/사진 비서: STT·요약·중복 정리 워크플로 미리보기
- 대화 비서에서 회의/사진 관련 요청을 하면 자동으로 오디오·폴더 선택 대화상자가 뜨고, 최근에 사용한 파일/폴더를 재사용할 수 있는 입력 폼이 함께 표시됩니다. 입력을 완료하면 진행 상태와 취소 버튼이 표시되어 장시간 작업을 모니터링할 수 있습니다.

## 5. 데이터 & 모델 관리

- `data/정답지/metadata.json`에 문서별 `"document_title"`, `"description"`, `"file_name"`을 기록하면 파이프라인이 메타데이터를 자동으로 병합합니다.
- 기본 문서 임베딩 모델은 `BAAI/bge-m3`이며, `DEFAULT_EMBED_MODEL`(infopilot CLI `--embedding-model`)로 다른 SentenceTransformer를 쉽게 대체할 수 있습니다.
- SentenceTransformer 모델을 `models/sentence_transformers/` 아래에 복사하면 CLI가 `HF_HOME`, `SENTENCE_TRANSFORMERS_HOME`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`을 자동 설정하여 오프라인에서 임베딩을 로드합니다.

## 6. 유지 보수

1. **테스트**  
   ```bash
   pytest -q
   ```

2. **파이프라인 재학습** (데이터 스키마 변경 시)  
   `scan` → `train` → `chat` 순으로 재실행

3. **대화 엔진 갱신** (모델/코퍼스 업데이트 후)  
   `infopilot.py run chat --cache data/cache`로 FAISS 인덱스를 갱신

4. **Git 워크플로**  
   ```
   git status
   git add <files>
   git commit -m "설명"
   git push origin <branch>
   ```

## 7. 추가 문서

- `docs/agents/`: 회의 비서 등 에이전트 설계·운영 가이드
- `docs/architecture/`: 시스템 구성, 데이터/모듈 상호작용
- `docs/guides/`: 로컬 LLM, 회의 모델 등 실사용 가이드
- `docs/plan/`: 로드맵 v3 정렬, 캐시/정책 전략, 테스트 체크리스트
- `docs/process/`: 운영/프로세스 정리, 체크리스트
- `docs/research/`: 실험 기록 및 벤치마크
- `docs/roadmap/`: 기능 계획 및 우선순위
- `docs/ux/`: UX 개선안, 피드백 로그

필요한 세부 가이드는 각 하위 디렉터리의 README 또는 문서를 참고하세요.
