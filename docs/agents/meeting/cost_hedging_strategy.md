# 비용 헷징 전략

회의 비서 파이프라인은 STT, 요약 LLM, 저장/인덱싱 단계에서 비용이 발생할 수 있습니다. 아래 전략을 통해 기본 동작은 로컬 우선으로 유지하고, 필요 시에만 클라우드 비용을 사용하도록 조정합니다.

## 비용 발생 포인트
- **STT**: Whisper/Faster-Whisper(로컬) vs. Google STT·Naver CLOVA 등 클라우드 API.
- **LLM**: Groq/OpenAI/Anthropic 등 원격 API는 토큰 과금, 로컬 모델은 GPU·전력 비용.
- **저장/인덱싱**: 로컬 FAISS/Chroma는 무료, Pinecone·Milvus 등 관리형 서비스는 유료.

## 비용 절감 전략
1. **로컬 우선(Local-first)**  
   - Whisper/Faster-Whisper를 기본으로 사용해 STT 비용을 0으로 유지합니다.  
   - Llama 3 8B, Mistral 7B 등 로컬 모델을 우선 사용하고, 고성능이 필요할 때만 클라우드 API를 호출합니다.  
   - 스마트 폴더 정책이나 `MeetingJobConfig`로 예외 조건을 명시합니다.

2. **캐싱 활용**  
   - 동일 음성은 STT 결과를 캐싱(`transcript.json`), 동일 질문은 요약/QA 캐싱(`cache_dir`)으로 중복 토큰 사용을 줄입니다.  
   - FAQ나 반복되는 회의 요약은 산출물을 보존해 재사용합니다.

3. **모델 크기 자동 분기**  
   - `token_length` 등 입력 특성에 따라 소형(로컬) ↔ 대형(클라우드) 모델을 자동 선택합니다.  
   - 정책 예시: `if token_length < 1000 → local`, 그 외에는 클라우드 호출.

4. **사용자 등급화**  
   - 무료 구독자는 로컬 모드만 허용하고, 프리미엄 사용자는 Groq/OpenAI API를 사용하도록 차등화합니다.  
   - 비용을 수익원(유료 계정)과 연결해 자연스럽게 헷징합니다.

5. **예산 모니터링**  
   - API 호출 횟수/토큰 수를 모니터링하고 월 한도를 초과하면 로컬 모드로 자동 전환합니다.  
   - UI/CLI에서 “예산 초과 → 로컬로 전환” 안내 메시지를 제공합니다.

## 정책 예시

```json
{
  "budget_mode": "local_first",
  "max_api_calls": 1000,
  "cache": true
}
```

스마트 폴더 정책이나 `configs/meeting_agent.yaml`에 위와 같은 옵션을 추가하면 파이프라인 전체가 비용-우선순위에 맞춰 동작합니다.
