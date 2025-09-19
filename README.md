# InfoPilot 문서 요약/검색 도구

## 설치
1. 가상환경을 생성하고 활성화합니다.
2. 의존성을 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```
3. 선택 기능(번역, 추가 포맷 추출)을 위해 다음 패키지를 권장합니다.
   ```bash
   pip install textract googletrans==4.0.0-rc1
   ```

> **참고**: `textract`는 시스템에 antiword/LibreOffice 등이 설치되어 있어야 `.doc`, `.ppt`, `.hwp` 추출이 활성화됩니다.

## 기본 워크플로우
```
python infopilot.py scan --path "<문서 루트>" --exclude venv
python infopilot.py train
python infopilot.py chat --rebuild
```

### 파이프라인 원커맨드
```
python infopilot.py pipeline --path "<문서 루트>"
```
`pipeline` 명령은 `scan → train → chat` 순으로 실행하며 학습 후 자동으로 인덱스를 재생성합니다.

## 주요 명령 및 옵션
| 명령 | 주요 옵션 | 설명 |
|------|-----------|------|
| `scan` | `--path`, `--exclude`, `--out` | 지정한 경로에서 지원 확장자 파일 목록을 수집합니다. 기본적으로 `.venv`, `venv`, `site-packages`, `AppData`, `Program Files`, `node_modules`, `__pycache__`, `~$*`, `*.tmp` 등을 자동으로 제외합니다. |
| `train` | `--scan_csv`, `--input`, `--corpus`, `--model`, `--index`, `--max_features` 등 | 스캔 결과 CSV를 읽어 텍스트를 추출하고 TF-IDF+SVD+MiniBatchKMeans 모델을 학습합니다. `--input`은 `--scan_csv`의 별칭, `--index`는 `--model`의 별칭입니다. |
| `chat` | `--model`, `--corpus`, `--cache`, `--topk`, `--rebuild` | 학습된 모델과 코퍼스를 로드하여 대화형 검색을 수행합니다. `--rebuild`를 사용하면 캐시된 인덱스를 무시하고 현재 코퍼스로 재생성합니다. |
| `pipeline` | `--path`, `--exclude`, `--scan-out`, `--corpus`, `--model`, `--cache`, `--topk` 등 | 스캔 → 학습 → 챗을 순차 실행합니다. |

## 추출 품질 및 실패 로그
- 다양한 추출기(예: PDF `pdfplumber → PyPDF2 → pdfminer → textract`)를 확장자별 레지스트리로 관리하여 가능한 경우 자동 폴백합니다.
- 추출에 실패하거나 비어 있는 텍스트는 `data/extract_failures.csv`에 `UTF-8-SIG` 인코딩으로 기록됩니다.
- 학습 가능한 문서 비율이 60% 미만이면 데이터 품질을 점검하세요.

## Windows PowerShell 팁
- 인덱스 캐시 삭제는 `Remove-Item -Recurse -Force index_cache` 명령을 사용하세요.
- 경로에 공백이 있을 경우 반드시 따옴표로 감싸고, UTF-8-SIG 인코딩을 기본으로 사용합니다.

## 테스트
- `tests/` 디렉터리에는 기본 제외 규칙과 추출 파이프라인을 검증하는 단위 테스트가 포함되어 있습니다.
- 샘플 문서는 테스트 실행 시 동적으로 생성되며 바이너리 파일을 저장하지 않습니다.
- 다음 명령으로 테스트를 실행할 수 있습니다.
  ```bash
  pytest
  ```
