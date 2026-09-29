# DECAT study dataset

`DECAT_Dataset_with_References.xlsx` is the current public data table for this study, published with release v1.0.3. It contains 1,626 experimental records in one worksheet.

| Column | Description |
| --- | --- |
| `NO.` | Original record identifier. Identifiers are preserved and contain gaps. |
| `Compounds` | Chemical substance name as recorded in the study dataset. |
| `log k` | Experimental log k value used by DECAT. |
| `pH` | pH value recorded for the measurement. |
| `Reference` | Source reference or database identifier, including DOI strings and NIST entries. |

## Provenance and missing references

This workbook presents the five study-data fields without molecular descriptors or fingerprint columns. Records retain their original order and identifiers. Experimental log k and pH values agree with the corresponding records in the  1,626-record dataset. 

The workbook is the public study-data download from v1.0.3 onward. The fingerprint-enriched CSV files under `core/locked_package/data/` and `deployment/artifacts/runtime_package/data/` remain the processed inputs required by the locked training and inference workflows. Those feature files and the model artifacts are unchanged by this data-table update.

Data reuse is subject to [LICENSE-DATA-MODEL-RESEARCH-USE.txt](../LICENSE-DATA-MODEL-RESEARCH-USE.txt) and the terms of the cited sources.
