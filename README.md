# Chemistry-Informed Dual-Expert Learning for Predicting Ozone-Pollutant Reaction Kinetics

DECAT includes training and prediction source, pretrained weights, the original experimental dataset, and a [Streamlit app](https://decat-ozone.streamlit.app/).

## Authors

Guosen Zhang<sup>a,b,1</sup>, Ninghan Wen<sup>a,1</sup>, Chaosen Yang<sup>b</sup>, Jinqi Jiang<sup>a,b</sup>, Haoran Chen<sup>b</sup>, Yiwen Tang<sup>a</sup>, Lei Wang<sup>a</sup>, Pengchao Xie<sup>b,*</sup>, Xueming Liu<sup>a,*</sup>

<sup>a</sup> School of Artificial Intelligence and Automation, Institute of Medical Equipment Science and Engineering, Huazhong University of Science and Technology, Wuhan 430074, China

<sup>b</sup> Institute of Artificial Intelligence, Hubei Key Laboratory of Multi-media Pollution Cooperative Control in Yangtze Basin, Key Laboratory of Water & Wastewater Treatment (MOHURD), Hubei Provincial Engineering Research Center for Water Quality Safety and Pollution Control, Huazhong University of Science and Technology, Wuhan 430074, China

<sup>1</sup> Guosen Zhang and Ninghan Wen contributed equally to this work.

<sup>*</sup> Corresponding authors: Xueming Liu (xm_liu@hust.edu.cn); Pengchao Xie (pengchao_xie@hust.edu.cn).

## Run

Use Python 3.13. From the repository root:

```shell
python -m pip install -r deployment/requirements.txt
python -m streamlit run deployment/streamlit_app.py
```

On Linux, install the system packages in `packages.txt`. For Streamlit Community Cloud, select Python 3.13 and `deployment/streamlit_app.py` as the main file.

## Train

Supply a CSV containing `NO.`, `SMILES`, and `category27_label`, with one row per original record. `NO.` must match the identifiers in the workbook. Labels use the model's class vocabulary in `core/locked_package/configs/preprocessing.json`; `CAS` is optional. Experimental values are read from the original workbook.

```shell
python core/locked_package/train.py --verify-only
python core/locked_package/train.py --inputs molecular_inputs.csv --output-root outputs/training
```

The program generates fingerprints and a deterministic split locally. Exact reproduction of the original training requires the original molecular structures and class assignments. Generated inputs, splits, and training outputs are excluded from Git.

## Files

- `data/DECAT_Dataset_with_References.xlsx`: 1,626 original records with `NO.`, `Compounds`, `log k`, `pH`, and `Reference`.
- `core/locked_package/src/decat/`: model, data processing, training, inference, chemistry, evaluation, interpretation, and plotting source.
- `core/locked_package/configs/`: model settings and fitted preprocessing parameters. `preprocessing.json` also contains the ordered training structures/names for applicability assessment and compound lookup information used by the website. These are public runtime resources, with no experimental target values or complete split lists.
- `deployment/artifacts/`: neural checkpoints, residual model, and ensemble configuration.
- `deployment/data/category_overrides.csv`: structure-to-class assignments needed by the predictor.

Archived releases are available on [Zenodo](https://doi.org/10.5281/zenodo.21922104).

Code is covered by the [MIT License](LICENSE-CODE-MIT.txt). Data and model weights are covered by the [research-use terms](LICENSE-DATA-MODEL-RESEARCH-USE.txt); third-party source materials retain their own terms.
