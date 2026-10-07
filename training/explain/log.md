# Explainability experiments

- `validate_xai.py` runs local sampled Shapley values, global SHAP permutation explanations, sentence-level LIME cross-checks, additivity checks, deletion/insertion comparisons, seed/background stability, and split-graph score parity on a reproducible sample of the 2024 test data.
- `integrated_gradients.py` is an independent, offline Captum check over token embeddings. Its PAD-token baseline and convergence delta must be reported with the output.
- Research-only dependencies are listed in `requirements.txt`; none are added to the API image.
- **2026-10-07:** scripts and API unit tests added. Model-based experiments have not run here because the non-git checkpoint, ONNX bundle and training parquet are absent. There are no XAI quality or latency claims yet.
