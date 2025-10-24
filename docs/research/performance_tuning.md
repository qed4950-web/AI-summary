# 성능 튜닝 가이드

개인화 가중치, 검색 속도, 벤치마크 흐름을 점검할 때 참고하는 설정 모음입니다.

## 벤치마크 실행

ANN(HNSW) 검색과 정확 검색(FAISS)을 비교하려면 다음 스크립트를 사용합니다.

```bash
source .venv/bin/activate
python -m scripts.benchmarks.ann_benchmark \
  --doc-count 20000 --queries 100 --top-k 10 \
  --ann-threshold 2000 --ef-search 128 --ef-construction 200 --ann-m 32 \
  --target-overlap 0.95 --target-p95 500 --output /tmp/ann.json
```

결과에는 평균/중앙/95퍼센타일 지연 시간, 정확 검색과의 겹침 비율, 속도 향상 배수가 포함됩니다. `--target-overlap`과 `--target-p95`를 설정하면 목표치에 미달할 때 즉시 종료하도록 만들 수 있습니다. 데이터 규모(`doc-count`, 차원 수 등)는 환경에 맞춰 조정하세요.

## CrossEncoder 조기 종료

- `EarlyStopConfig.score_threshold`: 이 값 미만의 배치 점수가 연속되면 학습을 중단합니다.
- `EarlyStopConfig.window_size`: 최근 배치 점수를 얼마나 묶어서 볼지 지정합니다(기본값은 배치 크기).
- `EarlyStopConfig.patience`: 연속으로 허용할 저점수 구간 개수입니다.  
  재현율이 떨어지면 `window_size`나 `patience`를 늘리고, 노이즈가 심하면 임계값을 낮추세요.

## 세션 개인화 가중치

`core/search/retriever.py`에는 다음 상수가 정의되어 있습니다.

| 상수 | 설명 | 기본값 |
| --- | --- | --- |
| `_SESSION_EXT_PREF_SCALE` | 확장자 선호도 가중치 | `0.05` |
| `_SESSION_OWNER_PREF_SCALE` | 소유자 선호도 가중치 | `0.04` |
| `_SESSION_CLICK_WEIGHT` | 클릭 반영 가중치 | `0.35` |
| `_SESSION_PIN_WEIGHT` | 고정 문서 가중치 | `0.6` |
| `_SESSION_LIKE_WEIGHT` | 좋아요 가중치 | `0.45` |
| `_SESSION_DISLIKE_WEIGHT` | 싫어요 가중치 | `-0.5` |
| `_SESSION_PREF_DECAY` | 갱신 시 감쇠율 | `0.85` |

값을 변경한 뒤에는 `pytest -m smoke`와 ANN 벤치마크를 다시 돌려 영향도를 검증하세요.

## 테스트 범위

- `tests/test_ann_and_reranker.py`: 조기 종료 로직, ANN 정확도, 세션 요약, 검색 훅 전달을 검증합니다.
- `tests/test_retriever_ext_filter.py`: 확장자/소유자 선호도와 메타데이터 필터를 확인합니다.

배포 전에는 `pytest -m full`로 전체 회귀를 권장합니다.

### 정확도/지연 시간 스위프

다양한 파라미터(`--ef-search`, `--ann-m`, `--rerank-min-score` 등)를 조합해 결과를 비교하려면 다음 패턴을 사용할 수 있습니다.

```bash
mkdir -p results/benchmarks
for ef in 64 96 128 160; do
  python -m scripts.benchmarks.ann_benchmark \
    --doc-count 20000 --queries 100 --top-k 10 --ann-threshold 2000 \
    --ef-search "$ef" --ef-construction 200 --ann-m 32 \
    --target-overlap 0.95 --target-p95 500 \
    --output results/benchmarks/ann_ef${ef}.json || true
done
```

저장한 JSON을 비교해 목표 값을 만족하는 최소 구성만 남기고, 선택한 설정은 저장소에 함께 기록해 회귀를 추적하세요.

## 정확도 평가

라벨이 있는 질의 집합이 있다면 다음 명령으로 P@K와 nDCG를 확인할 수 있습니다.

```bash
python -m scripts.benchmarks.accuracy_eval \
  --labels docs/research/benchmarks/fixtures/sample_labels.csv \
  --predictions docs/research/benchmarks/fixtures/sample_predictions.csv \
  --k 1 5 --target-p 0.8 --target-ndcg 0.7 || true
```

첫 번째 `k` 값에서 목표를 달성하지 못하면 종료 코드 1을 반환하므로 CI 회귀 검증에 활용 가능합니다. 라벨/예측 샘플은 `benchmarks/fixtures/`에 버전별로 보관하세요.
