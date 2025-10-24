# 문서 비서 아키텍처

문서 비서는 LNPChat(`core/conversation/lnp_chat.py`)을 중심으로 벡터 검색(`core/search/retriever.py`)과 정책 엔진(`core/data_pipeline/policies/engine.py`)을 결합해 동작합니다. UI/CLI에서 문서 검색을 요청하면 `DocumentAgent`가 아래 순서로 파이프라인을 실행합니다.

## 구성 요소

| 계층 | 주요 모듈 | 설명 |
| --- | --- | --- |
| 에이전트 래퍼 | `core/agents/document/agent.py` | `DocumentAgent`/`DocumentAgentConfig` 정의, LNPChat 초기화와 재색인을 제어 |
| 대화 엔진 | `core/conversation/lnp_chat.py` | 질의 전처리, 임베딩 검색, LLM 요약, 후처리를 담당 |
| 검색 계층 | `core/search/retriever.py` | SentenceTransformer 임베딩 + BM25 가중치 + 재랭커를 조합해 후보 문서를 반환 |
| 정책 엔진 | `core/data_pipeline/policies/engine.py` | 스마트 폴더 정책을 로드하고 적합한 문서만 노출 |
| 데이터 자원 | `data/corpus.parquet`, `data/topic_model.joblib`, `data/cache/` | 훈련된 임베딩/토픽 모델과 캐시(FAISS/JSONL) |

## 처리 흐름

1. **준비 단계**  
   `DocumentAgent.prepare()`가 호출되면 `LNPChat.ready(rebuild=...)`를 통해 벡터 인덱스를 확인하고 필요 시 재구축합니다.
2. **질의 접수**  
   `DocumentAgent.run()`이 `AgentRequest`를 받아 `LNPChat.ask()`에 질의를 전달합니다. 정책 스코프(`policy_scope`)와 스마트 폴더 규칙이 적용됩니다.
3. **검색/재랭킹**  
   `Retriever`가 top-k 문서를 찾고, 옵션에 따라 Cross-Encoder 재랭커가 점수를 보정합니다. `lexical_weight`가 지정되면 BM25 점수와 혼합합니다.
4. **LLM 요약**  
   요청에 따라 LLM(로컬 Ollama 또는 원격 API)을 호출해 응답 텍스트를 구성하고, 번역 옵션이 활성화되어 있으면 결과를 후처리합니다.
5. **응답 조합**  
   최종 응답은 `AgentResult`로 래핑되어 답변 본문, 제안 메시지, 히트 메타데이터(`path`, `score`, `preview`)를 포함합니다.

## 주의 사항

- 모델/코퍼스 파일이 누락되면 `DocumentAgent`는 `RuntimeError`를 발생시킵니다. 실행 전 `docs/agents/document/runbook.md`의 점검 절차를 따라 주세요.
- LLM 옵션(`llm_backend`, `llm_model`, `llm_host`)은 `.env` 또는 UI 설정에서 지정할 수 있으며 기본값은 `ollama` + `llama3`입니다.
- 정책 엔진을 사용하지 않을 경우 `PolicyEngine.empty()`를 넘기면 글로벌 스코프에서 동작합니다.
