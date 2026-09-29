# Data and code availability boundary

This package is a reproducibility/deployment bundle for the DECAT manuscript. It includes the locked 1,626-record working dataset, fixed split, model source, checkpoints, residual-RF artifact, Streamlit code, the ten external-validation records, and the derived 4,295-structure applicability-domain results.

The current public study dataset is [DECAT_Dataset_with_References.xlsx](data/DECAT_Dataset_with_References.xlsx), introduced in v1.0.3. It contains 1,626 records with the columns `NO.`, `Compounds`, `log k`, `pH`, and `Reference`. The [data dictionary](data/README.md) describes the fields and missing-reference markers. The corresponding fingerprint-enriched files are retained as processed model inputs for reproducibility.

Versioned releases are archived through the repository's Zenodo integration. The [all-versions DOI](https://doi.org/10.5281/zenodo.21922104) identifies the versioned archive; use the DOI of the particular release when citing a fixed snapshot. The v1.0.2 DOI, `10.5281/zenodo.21922105`, identifies the earlier release.

The external screening tables retain source identifiers, DOI metadata, and derived labels. Original third-party papers, supplementary PDFs, and other publisher files are not included. Users should obtain those sources from the cited publishers or repositories and follow their terms of use.

The public release repository for this package is [github.com/WenNinghan/DECAT](https://github.com/WenNinghan/DECAT). It contains the versioned checksum manifest and no third-party paper PDFs. Before citing or redistributing the package, users should verify that their intended use of the bundled data and model weights is compatible with the authors' permissions and the terms of the cited source records.
