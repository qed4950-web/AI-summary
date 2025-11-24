# webapp

Vite + React 프런트엔드로, 로컬 FastAPI 백엔드(`/api/search`)와 연동하는 경량 UI입니다. 오프라인 배포를 전제로 하여 모든 정적 자산은 번들에 포함됩니다.

## 기본 설정

```bash
cd webapp
cp .env.example .env         # 기본 API: http://127.0.0.1:8080

# 의존성 설치
npm install
```

## 개발 서버

```bash
npm run dev -- --host 127.0.0.1 --port 5173
```

## 프로덕션 빌드

```bash
npm run build
# 결과물: webapp/dist
```

## FastAPI 백엔드 예시

이미 리포에 있는 `scripts/api_server.py`를 실행합니다.

```bash
python scripts/api_server.py  # 기본 포트 8080
```

환경 변수 `VITE_API_BASE`로 다른 포트/호스트를 지정할 수 있습니다.

## 오프라인 배포

`npm install` 시 필요한 패키지만 다운로드되며, 이후에는 `npm run build`로 생성된 `dist/`를 Tauri/Electron 등으로 감싸거나 정적으로 서빙할 수 있습니다.

### Tauri 네이티브 패키징(선택)
- Rust toolchain + Cargo 필요 (`brew install rust` 등).
- 설치 후:
  ```bash
  cd webapp
  npm install
  npm run tauri:build   # 또는 npm run tauri:dev
  ```
  실행 시 내부에서 `python scripts/api_server.py`를 사이드카로 띄우도록 설정돼 있습니다(tauri.conf.json).
