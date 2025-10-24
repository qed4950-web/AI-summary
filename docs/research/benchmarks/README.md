# 벤치마크 자료

이 디렉터리는 과거 실험 스크립트와 샘플 라벨/예측 파일을 보관합니다.

- `fixtures/`: P@K, nDCG 계산에 사용할 수 있는 샘플 라벨/예측 CSV
- `accuracy_eval.py`, `ann_benchmark.py`: 초기 실험 코드 (보존용)

> 최신 실행 스크립트는 `scripts/benchmarks/`에 있습니다.  
> 문서화된 절차는 `docs/research/performance_tuning.md`를 참고하세요.

### 샘플 사용법

```bash
python -m scripts.benchmarks.accuracy_eval \
  --labels docs/research/benchmarks/fixtures/sample_labels.csv \
  --predictions docs/research/benchmarks/fixtures/sample_predictions.csv \
  --k 1 5
```

필요에 맞게 CSV를 교체하고, 결과 JSON은 `results/` 등 별도 디렉터리에 저장해 추적하세요.
