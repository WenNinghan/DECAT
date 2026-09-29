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

This workbook presents the five study-data fields without molecular descriptors or fingerprint columns. Records retain their original order and identifiers. Experimental log k and pH values agree with the corresponding records in the locked 1,626-record dataset. Reference values were transferred from the original `Ref` field by record identifier.

Source identifiers are present for 1,624 records. The original reference field is empty for `NO. 345` (`(CH3)2NCl`) and `NO. 1098` (`H2S`). Their cells read `Not provided in original dataset`; no citation has been inferred or assigned to those records.

The workbook is the public study-data download from v1.0.3 onward. The fingerprint-enriched CSV files under `core/locked_package/data/` and `deployment/artifacts/runtime_package/data/` remain the processed inputs required by the locked training and inference workflows. Those feature files and the model artifacts are unchanged by this data-table update.

Data reuse is subject to [LICENSE-DATA-MODEL-RESEARCH-USE.txt](../LICENSE-DATA-MODEL-RESEARCH-USE.txt) and the terms of the cited sources.
