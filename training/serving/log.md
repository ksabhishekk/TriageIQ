# Serving model exports

- `export_onnx.py` builds the production `model.onnx` bundle; the existing parity and latency results are recorded in `training/results/onnx_v3.json`.
- `export_xai_onnx.py` separately exports the text-vector and fusion-head graphs plus a fixed background sampled from the 2024 serving/test rows. It uses only text and intake-time feature columns, never labels, and writes files only after parity against `model.onnx` passes on 1,000 test complaints.
- **2026-10-07:** source added; not run in this checkout because the model checkpoint, serving bundle and training parquet are absent. Do not set `XAI_ENABLED=1` until the Mac run passes and API latency/RAM are checked on the target server.
