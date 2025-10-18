# AI-summary
AI 요약 소프트웨어를 구축하고 학습·조회 파이프라인을 돌리기 위한 저장소입니다.

## 프로젝트 개요
- `infopilot.py` : `scan → train → chat` 절차를 오케스트레이션하는 CLI 진입점
- `core/data_pipeline/filefinder.py` : 파일 시스템을 스캔해 후보 목록을 CSV로 저장
- `core/data_pipeline/pipeline.py` : 코퍼스 정제 및 토픽 모델 학습, Parquet/CSV 아티팩트 생성
- `core/search/retriever.py` / `core/conversation/lnp_chat.py` : 학습된 모델을 불러와 쿼리 검색 및 대화 인터페이스 제공
- `core/agents/meeting/` : 회의 비서(STT→요약) 파이프라인 초안과 설정이 위치
- `core/agents/photo/` : 사진 비서(태깅·중복 정리) MVP 골격과 설정 템플릿 제공
- `core/infra/` : 하이브리드 오프로딩/감사 로깅/모델 선택 유틸리티
- `data/`, `index_cache/` : 생성된 코퍼스와 캐시가 위치하는 디렉터리 (대용량 파일은 커밋 금지)

## 리포지토리 구조

```
AI-summary/
├─ core/          # 엔진, 파이프라인, 에이전트 구현
│  ├─ agents/     # 회의/검색/사진 등 도메인별 에이전트
│  ├─ conversation/
│  ├─ data_pipeline/
│  ├─ infra/
│  ├─ search/
│  ├─ utils/
│  └─ legacy/     # 이전 백엔드/패키지(backend, core, src)
├─ ui/            # 데스크톱 UI (CustomTkinter)
│  ├─ app.py
│  └─ legacy/web_frontend/   # React 기반 기존 UI
├─ data/          # 실행 중 생성되는 캐시/산출물 (Git 무시)
│  ├─ ami_outputs/
│  ├─ artifacts/
│  ├─ index_cache/
│  ├─ build/, dist/
│  └─ tools/
├─ models/        # 모델 체크포인트 캐시 (Git 무시)
├─ scripts/       # CLI·런처·빌드 스크립트
│  ├─ infopilot.py
│  ├─ launch_desktop.py
│  ├─ packaging/
│  └─ …
├─ docs/          # 설계·로드맵·벤치마크 등 문서
│  ├─ benchmarks/
│  └─ notes/
├─ tests/
└─ README.md
```

`.gitignore`에는 `data/` 전체와 `models/` 폴더가 포함되어 있으므로, `ami_outputs`, `artifacts`, `index_cache`, `build`, `dist`, 모델 체크포인트 같은 대용량 산출물이 Git에 올라가지 않습니다. 필요하다면 `data/.gitkeep` 등 빈 파일을 추가해 디렉터리 구조만 유지할 수 있습니다.

## 핵심 명령
| 단계 | 명령 | 설명 |
| --- | --- | --- |
| 스캔 | `python scripts/infopilot.py scan --out data/found_files.csv` | 새/변경 파일 메타데이터 수집 |
| 학습 | `python scripts/infopilot.py train --scan_csv data/found_files.csv --corpus data/corpus.parquet` | 코퍼스 및 토픽 모델 생성 |
| 대화 | `python scripts/infopilot.py chat --model data/corpus/topic_model.joblib --corpus data/corpus.parquet --cache data/index_cache` | 의미 기반 검색 챗봇 실행 (`--translate`, `--show-translation`, `--lexical-weight` 옵션 지원) |
| 감시 | `python scripts/infopilot.py watch --cache data/index_cache --corpus data/corpus.parquet --model data/corpus/topic_model.joblib` | 파일 변경을 감지해 코퍼스·FAISS 인덱스를 실시간으로 증분 갱신 |
| 예약 | `python scripts/infopilot.py schedule --policy core/config/smart_folders.json` | 스마트 폴더 정책의 예약 스캔/학습을 실행하고 모니터링 |

파이프라인실행
python scripts/infopilot.py pipeline --corpus data/corpus.parquet --model data/corpus/topic_model.joblib

## 데스크톱 UI 실행
- `python scripts/launch_desktop.py` (또는 개발 환경에서는 `python ui/app.py`)로 데스크톱 런처를 실행하면 하나의 창에서 주요 기능을 모두 사용할 수 있습니다.
  - 홈 대시보드: 코퍼스/모델 준비 상태 확인 및 빠른 이동 카드
  - 지식·검색 비서: 의미 검색과 필터, 결과 미리보기
  - 전체 학습 / 증분 업데이트: 기존 CLI 파이프라인을 GUI에서 실행
  - 회의 비서 MVP: 오디오·전사 파일 요약, 액션 아이템·결정 사항 추출, 결과 폴더 저장
  - 사진 비서 MVP: 폴더 스캔, 중복 그룹·베스트샷 추천, `photo_report.json` 생성
- 회의 비서를 사용하려면 `pip install -r requirements_win313.txt torch`로 필수 의존성을 준비하고, 다음 환경 변수를 설정하세요.
  - `MEETING_OUTPUT_DIR`: 회의별 산출물을 저장할 루트 경로 (예: `C:\python\github\AI-summary\ami_outputs`).
  - `MEETING_ANALYTICS_DIR`: 대시보드와 재학습 큐를 저장할 경로. 지정하지 않으면 산출물 폴더 내 `analytics/`를 사용합니다.
  - `MEETING_SUMMARY_MODEL`: 요약 기본 모델 경로(예: fine-tuned KoBART 체크포인트). 지정하지 않으면 기본값 `gogamza/kobart-base-v2`.
  - 선택 옵션: `MEETING_RETRAIN_OUTPUT_DIR`, `MEETING_AUDIT_LOG`, `MEETING_MASK_PII`, `MEETING_SAVE_TRANSCRIPT` 등.
- 런처 실행 후 **회의 비서** 탭에서 작업하면 `summary.json`, `action_items.json`, `meeting.ics`, `analytics/dashboard.json` 등이 생성되고, UI와 `/api/meeting/dashboard`에서 결과를 확인할 수 있습니다.
- LNPChat에서 로컬 LLM(예: Ollama)을 사용하려면 `docs/guides/local_llm.md`를 참고하세요.
- 재학습을 돌리고 싶다면 아래 순서로 진행하세요.
  1. 회의 실행 후 각 회의 디렉터리에 생성된 `analytics/training_queue.jsonl`을 통합합니다.
     ```powershell
     python -m core.agents.meeting.retraining_sync \
       --source-root "C:\\python\\github\\AI-summary\\ami_outputs" \
       --output-dir "C:\\python\\github\\AI-summary\\analytics"
     ```
  2. 미세조정을 실행합니다 (`--max-runs 0`은 큐가 빌 때까지 반복, `--watch`는 주기적으로 큐를 감시).
     ```powershell
     python -m core.agents.meeting.retraining_runner \
       --mode finetune \
       --analytics-dir "C:\\python\\github\\AI-summary\\analytics" \
       --dataset-root "C:\\python\\github\\AI-summary\\ami_outputs" \
       --output-root "C:\\python\\github\\AI-summary\\artifacts\\retraining" \
       --base-model "gogamza/kobart-base-v2" \
       --max-runs 0
     ```
     실행 후 각 체크포인트 폴더에 `run_summary.json`이 저장되어 최종 평가 loss 등을 확인할 수 있습니다.
  (이전 FastAPI/React 기반 서비스는 제거했습니다. 데스크톱 앱에서 회의 분석 버튼을 통해 동일 지표를 확인할 수 있습니다.)
- PyInstaller로 Windows용 실행 파일을 만들고 싶다면 아래 명령을 사용하세요.
  ```powershell
  python -m pip install --upgrade pip
  pip install -r requirements_win313.txt pyinstaller
  powershell -ExecutionPolicy Bypass -File scripts\build_desktop_ui.ps1 -OutputName InfoPilotDesktop
  ```
  설치용 ZIP까지 만들고 싶다면 `-CreateZip` 스위치를 추가하세요.
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts\build_desktop_ui.ps1 -OutputName InfoPilotDesktop -CreateZip
  ```
  생성된 `dist\InfoPilotDesktop\InfoPilotDesktop.exe` 또는 `dist\InfoPilotDesktop.zip`을 배포하면 사용자는 압축만 풀고 exe를 실행하면 됩니다.

### 스마트 폴더 정책
- 모든 명령은 `--policy` 옵션으로 스마트 폴더 정책(JSON) 경로를 지정할 수 있습니다. 기본값은 `config/smart_folders.json`이며, 비활성화하려면 `--policy none`을 사용하세요.
- `chat` 명령은 추가로 `--scope` 옵션을 제공합니다. `auto`(기본)는 정책이 존재할 때만 적용, `policy`는 강제 적용, `global`은 정책을 무시하고 전역 검색을 수행합니다.
- `schedule` 명령을 사용하면 `indexing.mode = scheduled`로 설정된 스마트 폴더를 주기적으로 스캔·학습해 별도 산출물 디렉터리에 적재할 수 있습니다 (`--output-root`로 경로 지정).
- 정책 스키마는 `core/data_pipeline/policies/schema/smart_folder_policy.schema.json`, 예시는 `core/data_pipeline/policies/examples/smart_folder_policy_sample.json`에서 확인할 수 있습니다.
- 정책이 활성화되면 Knowledge & Search 에이전트(`knowledge_search`)가 허용된 폴더만 스캔·학습·워치 대상으로 처리되며, 채팅 결과에서도 정책으로 제외된 문서 수를 안내합니다.

## 업데이트 및 검증 절차

### 1. 사전 준비
- Python 3.9 이상 설치 여부를 확인합니다 (`python --version` 또는 `python3 --version`).
- 모든 명령은 저장소 루트에서 실행하는 것을 권장합니다.

### 2. 가상환경 생성 및 의존성 설치

#### 공통 (권장)
```bash
python -m venv .venv
```

#### macOS / Linux
```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows (Command Prompt)
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> 기존 가상환경이 있다면 생성 단계는 건너뛰고 활성화만 진행하세요.

### 3. 업데이트 검증 플로우
1. **테스트 실행**
   ```bash
   pytest -q
   ```
   Windows에서 `pytest`가 PATH에 없다면 `py -m pytest -q`를 사용하세요.
2. **파이프라인 재구축(데이터 로직 변경 시)**
   ```bash
   python infopilot.py scan --out data/found_files.csv
   python infopilot.py train --scan_csv data/found_files.csv --corpus data/corpus.parquet
   ```
3. **챗 캐시 새로고침(코퍼스/모델 변경 시)**
   ```bash
   python infopilot.py chat --model data/topic_model.joblib --corpus data/corpus.parquet --cache index_cache
   ```
   번역을 강제하려면 `--translate` 옵션을 추가하세요.
4. **증분 감시로 인덱스 유지(선택 사항)**
   ```bash
   python infopilot.py watch --cache index_cache --corpus data/corpus.parquet --model data/topic_model.joblib
   ```

### 4. OS 별 참고 사항
- **macOS**: `python3` 명령 사용을 권장하며, C 확장 빌드 오류 시 `xcode-select --install`로 Command Line Tools를 설치하세요.
- **Windows**: 64비트 Python을 사용하고, 의존성 컴파일 실패 시 Microsoft C++ Build Tools 설치를 확인하세요.
- **WSL / Linux**: `sudo apt-get install build-essential libsnappy-dev`로 빌드 도구를 준비하고, 환경에 따라 `python3` 명령을 사용하세요.

### 5. 검증 후 마무리
- 필요 시 `deactivate`로 가상환경을 종료합니다.
- 변경 사항을 스테이징하고 커밋/푸시합니다.



## OS별 설치 및 실행 가이드
### macOS
1. Homebrew 등으로 Python 3.9 이상이 설치되어 있는지 확인 (`python3 --version`).
2. 가상환경 생성 및 활성화:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. 의존성 설치:
   ```bash
   python3 -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Parquet 의존성(예: `fastparquet`) 빌드 에러가 나면 Xcode Command Line Tools를 설치하세요.
   ```bash
   xcode-select --install
   ```
5. 필요 시 명령을 실행합니다 (예: `pytest -q`, `python infopilot.py train ...`).

### Linux / WSL
1. 필수 빌드 도구 준비(Parquet 가속을 위해 권장):
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential libsnappy-dev
   ```
2. 가상환경 생성 및 활성화:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. 의존성 설치:
   ```bash
   python3 -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *시스템 Python을 그대로 사용할 경우 Debian/Ubuntu는 `--break-system-packages` 옵션이 필요할 수 있습니다.*
4. 테스트/학습/대화를 필요한 순서대로 실행합니다.

### Windows (PowerShell)
1. 64비트 Python 3.9 이상 설치 여부 확인 (`python --version`).
단계별 절차 (윈도우 PowerShell 기준)

기존 venv 삭제

Remove-Item .venv -Recurse -Force


 2. 가상환경 생성 및 활성화: 
   ```powershell
    python -m venv .venv
   .\.venv\Scripts\Activate.ps1 
   ```
3. 의존성 설치:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. `fastparquet` 설치가 실패하면 Microsoft C++ Build Tools를 설치하거나 최신 휠이 포함된 Python 버전을 사용하세요.
5. 이후 `pytest -q`, `python infopilot.py scan ...` 등 필요한 명령을 순차 실행합니다.

#### Windows 실행 파일(.exe) 빌드
PowerShell에서 아래 순서로 실행하면 `dist/InfoPilotLauncher.exe`가 생성됩니다.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt pyinstaller
powershell -ExecutionPolicy Bypass -File scripts\build_windows_exe.ps1
```

빌드된 파일을 더블 클릭하면 간단한 콘솔 메뉴가 열리고, 파이프라인 실행(옵션: 번역, 루트 경로 입력) 후 Chat 자동 실행 또는 Chat만 단독 실행을 선택할 수 있습니다.

### Windows (Command Prompt)
1. 가상환경 생성 및 활성화:
   ```cmd
   python -m venv .venv
   .\.venv\Scripts\activate.bat
   ```
2. 의존성 설치:
   ```cmd
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. 명령 실행 예시:
   ```cmd
   pytest -q
   python infopilot.py chat --model data\topic_model.joblib --corpus data\corpus.parquet --cache index_cache --translate --show-translation
   ```

## 테스트 및 검증

### Benchmark & Evaluation
1. ANN vs 정확 검색 벤치마크
   ```bash
   source .venv/bin/activate
   python benchmarks/ann_benchmark.py \
     --doc-count 20000 --queries 100 --top-k 10 --ann-threshold 2000 \
     --ef-search 128 --ef-construction 200 --ann-m 32 \
     --target-overlap 0.95 --target-p95 500 --output benchmarks/results/ann_run.json
   ```
   - `overlap@k`(정확 검색과의 겹침)과 `ann_ms.p95`가 목표에 못 미치면 종료 코드가 1이므로 CI 회귀 검증에 사용할 수 있습니다.
2. 정확도 지표(P@K, nDCG@K) 측정
   ```bash
   python benchmarks/accuracy_eval.py \
     --labels benchmarks/fixtures/sample_labels.csv \
     --predictions benchmarks/fixtures/sample_predictions.csv \
     --k 1 5 --target-p 0.8 --target-ndcg 0.7 --output benchmarks/results/accuracy.json
   ```
   - 실측 라벨/예측 CSV로 바꿔 실행하고, 목표치를 벗어나면 비정상 종료로 회귀를 잡아낼 수 있습니다.
3. 프런트엔드 스모크
   ```bash
   pytest -m smoke
   ```
   - 세션/ANN 경로 관련 테스트가 포함되어 프런트 버튼-API 훅이 정상 동작하는지 재확인합니다.

### 새 UI 이벤트 로그 확인
로컬에서 카드 버튼을 클릭하면 `localStorage`의 `infopilot_ui_events` 키에 성공/실패 이벤트가 쌓입니다. 콘솔에서
```js
JSON.parse(localStorage.getItem('infopilot_ui_events') || '[]')
```
을 실행해 API 전송 누락 여부를 빠르게 점검하세요.

- 모든 플랫폼에서 `pytest -q`로 스모크 + 풀 테스트를 한 번에 수행하세요.
- 빠른 확인만 필요하면 `pytest -q -m smoke`, 전체 회귀만 재실행하려면 `pytest -q -m full`을 사용할 수 있습니다.
- `tests/conftest.py`가 런타임에 필요한 `JOBLIB_MULTIPROCESSING=0`과 임시 폴더 설정을 제공하므로 추가 환경 변수 설정이 필요 없습니다.
- 데이터 파이프라인 변경 시 `scan → train`을 재실행한 뒤 캐시 무결성을 확인하고, `chat`에서 결과를 수동 검증하세요.

## 문제 해결 팁
- Parquet 저장 실패 시 `requirements.txt`에 포함된 `fastparquet`가 제대로 설치되었는지 확인하고, 빌드 도구나 네트워크 제한을 점검합니다. 필요한 경우 `pyarrow`와 같은 대체 엔진을 수동으로 설치해도 됩니다.
- 검색은 기본적으로 다국어 Sentence-BERT 의미 검색으로 동작합니다. 필요 시 `--lexical-weight`를 0 초과 값으로 지정해 BM25 보조 가중치를 추가하세요.
- 번역이 필요하면 `--translate`(질문 번역) 또는 `--show-translation`(결과 미리보기 번역) 옵션을 추가하고, 오류 발생 시 `deep-translator` 패키지와 외부 네트워크 접근성을 점검하세요.
- 실시간 반영이 필요하면 `python infopilot.py watch ...`로 watcher를 띄워 새/수정된 파일을 자동 추출·임베딩하세요.

깃허브 푸쉬방법

1. Git 상태 확인
git status


→ 수정된 파일/추가된 파일 목록 확인

2. 변경 파일 스테이징
git add -A


(전체 변경 사항을 스테이징)

3. 커밋 생성
git commit -m "파이프라인 실행 코드 및 테스트 통과 후 정리"


→ 메시지는 원하는 걸로 자유롭게 작성 가능

4. 원격 브랜치 확인
git branch -vv


→ 현재 작업 브랜치(main, test1 등)와 연결된 원격(remote/origin) 확인

5. 푸시
git push origin main


만약 현재 브랜치가 main이 아니라면, 예를 들어 test1이라면:

git push origin test1

6. (옵션) 강제 푸시

히스토리가 꼬였거나, 기존 커밋을 덮어써야 하는 경우만:

git push origin main --force


⚠️ 하지만 협업 환경에서는 권장하지 않습니다. 본인만 쓰는 저장소일 때만 사용하세요.
>>>>>>> recover-mywork
