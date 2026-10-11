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

## Scheduled public-app activation

The [Activate DECAT app workflow](https://github.com/WenNinghan/DECAT/actions/workflows/activate-app.yml) visits <https://decat-ozone.streamlit.app/> every six hours (02:17, 08:17, 14:17 and 20:17 Beijing time). It also supports **Run workflow** for an immediate manual check. Execution happens on GitHub runners and does not require the author's computer to stay on.

The independent `maintenance/activate_app.py` script opens an anonymous Chromium session, clicks the official wake-up button if present, and waits for the DECAT heading, chemical-name input and prediction button in the actual app frame. It does not run predictions. An HTTP 200 response alone is not counted as success. Each run saves a JSON report and screenshots as a seven-day Actions artifact; an unsuccessful check exits nonzero.

To run it yourself in a separate Python environment:

```shell
python -m pip install -r maintenance/requirements.txt
python -m playwright install --only-shell chromium
python maintenance/activate_app.py --output-dir activation-results
```

On Linux, use `python -m playwright install --with-deps --only-shell chromium` to install the browser's system dependencies as well. These maintenance dependencies are separate from the prediction app's pinned environment.

Streamlit's [current hibernation rule](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#app-hibernation) is 12 hours without traffic. This workflow is a best-effort visit and wake-up check, not an uptime guarantee. GitHub scheduled jobs can be delayed or dropped, and [public repositories with no activity for 60 days have their schedules disabled](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule). Re-enable the workflow from Actions if that happens. To stop the visits, select **Disable workflow** from its Actions menu.

## Local launch

Run from the release root:

```powershell
pwsh -File .\deployment\scripts\run_streamlit.ps1 -Python python
```

The first model load reconstructs training-only scalers and the 914 selected fingerprint bits; it does not fit neural weights or retrain any model. The platform uses the weighted Top-3 validation ensemble followed by the serialized `nn_plus_res_rf` residual model.
