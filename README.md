# DECAT: dual-expert chemistry-aware Transformer

This release packages the locked DECAT route used for the manuscript together with the Streamlit deployment and the seed-242 applicability-domain screen.

## Contents

- `core/locked_package/`: the public, portable copy of the locked reproduction package for the 1,626-record, 25-class dataset. The original locked source package is preserved separately by the authors.
- `deployment/`: self-contained Streamlit platform, Top-3 validation-ranked checkpoints, serialized `nn_plus_res_rf` residual model, runtime code, the 1,626-record reference data, and the ten published external-validation cases.
- `screening/`: seed-242 AD script, the 4,295-structure input library, the 11,654-row source table, and the resulting unique-structure and row-level labels.

No model is retrained by the deployment or screening commands. The original third-party paper PDFs are intentionally not included.

The public copy has been sanitized for redistribution: machine-specific absolute paths, Python bytecode caches, and local runtime references were removed. Optional legacy figure-rendering scripts accept package-relative defaults and can be redirected with environment variables such as `DECAT_ASSET_ROOT`, `DECAT_REFERENCE_SOURCE_ROOT`, `DECAT_FIGURE_STYLE_DIR`, and `DECAT_FONT_DIR`.

## Locked inference route

- Dataset: 1,626 records and 25 reaction classes.
- Fixed split: 1,240 train / 186 validation / 200 test; seed 242.
- Morgan fingerprint: radius 2, 3,147 bits; RF-ranked top 914 bits.
- Transformer: dual-expert, two layers, four heads, Post-LN.
- Prediction: weighted ensemble of `top1_epoch33.pth`, `top2_epoch31.pth`, and `top3_epoch21.pth`, followed by the serialized `nn_plus_res_rf` residual correction.
- Reference test result: R² = 0.828640; RMSE = 1.162404.

## Run the platform

From the release root:

```powershell
pwsh -File .\deployment\scripts\run_streamlit.ps1 -Python python
```

Alternatively:

```powershell
streamlit run .\deployment\streamlit_app.py
```

The platform resolves all model and data assets relative to the release directory. Internet access is only needed for optional PubChem/CACTUS lookups when a compound is not already present in the bundled reference/external tables.

## Run or verify the AD screen

The bundled screening result contains 4,295 unique structures and 11,654 source rows. The locked seed-242 unique-structure summary is:

- In AD: 3,423 (79.7%)
- Borderline: 699
- Out of AD: 51
- Out of DECAT scope: 82
- pH outside training range: 40

Regenerate the result without training:

```powershell
pwsh -File .\screening\scripts\run_ad_screening.ps1 -Python python
```

The script uses LOO q95 distance `0.6428571428571428`, q99 distance `0.8`, and training pH range `[1.0, 12.0]`.

## Verify deployment without training

```powershell
python .\deployment\scripts\verify_deployment.py
```

The verification loads the three checkpoints and residual model, performs one inference, and checks that the bundled screening table has exactly 4,295 unique rows.

## Data and redistribution boundary

The package contains the supplied DECAT dataset, model weights, derived predictions, external-validation table, and derived AD labels. The original third-party source papers/PDFs are not redistributed here; consult the citation and source metadata stored in the bundled tables for provenance. Users must verify that their intended use and redistribution of the bundled data and model artifacts are compatible with the authors' permissions and the cited source terms.
