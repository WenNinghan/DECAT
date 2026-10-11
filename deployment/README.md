# Streamlit deployment module

`streamlit_app.py` is the self-contained single-compound platform. It resolves the locked reference dataset, runtime source, Top-3 checkpoints, residual-RF artifact, external validation table, and seed-242 AD result table relative to this release directory.

## Streamlit Community Cloud

- Repository: `WenNinghan/DECAT`
- Branch: `main`
- Main file: `deployment/streamlit_app.py`
- **Advanced settings > Python version: `3.13`**
- App URL: `decat-ozone` (when available)

The requirements are pinned to the validated model environment. The Linux x86_64 / Python 3.13 target uses an official CPU-only PyTorch wheel; `rdkit` replaces the obsolete `rdkit-pypi` distribution. Keep `scikit-learn==1.7.2` to match the serialized residual model.

The repository-root `packages.txt` supplies `libxrender1` and `libxext6` for RDKit's Cairo drawing backend; these packages also install `libx11-6` as a dependency. These Linux shared libraries are not included in the RDKit wheel and are needed when importing `rdkit.Chem.Draw`. Community Cloud installs them with apt before starting the app.

Do not deploy this environment with Python 3.14. If an existing deployment uses 3.14, Community Cloud requires deleting that failed app instance and redeploying it with Python 3.13 selected in Advanced settings. Reuse the same repository, entrypoint, and custom subdomain. Rebooting the existing instance does not change its Python version. See the [Community Cloud Python-version instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python).

## Local launch

Run from the release root:

```powershell
pwsh -File .\deployment\scripts\run_streamlit.ps1 -Python python
```

The first model load reconstructs training-only scalers and the 914 selected fingerprint bits; it does not fit neural weights or retrain any model. The platform uses the weighted Top-3 validation ensemble followed by the serialized `nn_plus_res_rf` residual model.
