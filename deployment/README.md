# Streamlit deployment module

`streamlit_app.py` is the self-contained single-compound platform. It resolves the locked reference dataset, runtime source, Top-3 checkpoints, residual-RF artifact, external validation table, and seed-242 AD result table relative to this release directory.

Run from the release root:

```powershell
pwsh -File .\deployment\scripts\run_streamlit.ps1 -Python python
```

The first model load reconstructs training-only scalers and the 914 selected fingerprint bits; it does not fit neural weights or retrain any model. The platform uses the weighted Top-3 validation ensemble followed by the serialized `nn_plus_res_rf` residual model.
