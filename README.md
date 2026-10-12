# Chemistry-Informed Dual-Expert Learning for Predicting Ozone-Pollutant Reaction Kinetics

DECAT provides model training code, pretrained weights, and a Streamlit app for predicting ozone-pollutant reaction rate constants. [Open the public app](https://decat-ozone.streamlit.app/).

## Run

Use Python 3.13. From the repository root:

```shell
python -m pip install -r deployment/requirements.txt
python -m streamlit run deployment/streamlit_app.py
```

On Linux, install the system packages listed in `packages.txt`. For Streamlit Community Cloud, select Python 3.13 and `deployment/streamlit_app.py` as the main file.

## Train

Use the same Python environment and run from the repository root:

```shell
python core/locked_package/train.py --verify-only
python core/locked_package/train.py
```

Use `--output-root <directory>` to choose where local training outputs are saved. On Windows, `core/locked_package/scripts/train.ps1` provides the same training and `-VerifyOnly` options.

The fixed training configuration is in `core/locked_package/configs/decat.json`, with runtime defaults in `configs/runtime.json`. The source package is `core/locked_package/src/decat/`:

- `config.py` and `runtime.py`: settings, reproducibility, and run state.
- `chemistry.py` and `data/`: chemical categories, features, datasets, and split loading.
- `models/`: attention layers, expert networks, and the shared DECAT model.
- `training/`: losses, epoch execution, ensembling, selection protocol, and workflow.
- `inference.py`, `applicability.py`, and `evaluation.py`: prediction, applicability checks, and metrics.
- `interpretation/`, `plotting/`, and `experiments/`: explanations, figures, and optional experiments.

The app uses this shared package and the pretrained weights in `deployment/artifacts/`.

## Data and licenses

The [study dataset](data/DECAT_Dataset_with_References.xlsx) contains 1,626 records with `NO.`, `Compounds`, `log k`, `pH`, and `Reference` fields. Source references remain in the workbook. The processed [feature data](core/locked_package/data/decat_features_1626.csv) and [fixed split](core/locked_package/data/split.json) are the training and inference inputs.

Code is covered by the [MIT License](LICENSE-CODE-MIT.txt). Data and model weights are covered by the existing [research-use terms](LICENSE-DATA-MODEL-RESEARCH-USE.txt); third-party source materials retain their own terms.
