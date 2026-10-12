"""Join original measurements with explicitly supplied molecular training inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from ..input_chemistry import LOCKED_CATEGORY27_OPTIONS


WORKBOOK_COLUMNS = ("NO.", "Compounds", "log k", "pH", "Reference")
INPUT_COLUMNS = ("NO.", "SMILES", "category27_label")


def _record_ids(values: pd.Series, source: str) -> pd.Series:
    """Validate integer record identifiers before joining either input table."""
    numeric = pd.to_numeric(values, errors="raise")
    if (
        numeric.isna().any()
        or not np.isfinite(numeric.to_numpy(dtype=float)).all()
        or (numeric % 1 != 0).any()
    ):
        raise ValueError(f"{source}: NO. must contain finite integer identifiers.")
    identifiers = numeric.astype("int64")
    if identifiers.duplicated().any():
        duplicates = identifiers[identifiers.duplicated(keep=False)].unique().tolist()
        raise ValueError(f"{source}: duplicate NO. identifiers: {duplicates}")
    return identifiers


def load_training_inputs(workbook_path: Path, inputs_path: Path) -> pd.DataFrame:
    """Preserve workbook measurements/order and attach validated structures/classes.

    The supplied UTF-8 CSV contains NO., SMILES, category27_label and optionally
    CAS. It must cover every workbook record exactly once. Measurements can only
    come from the original workbook; the supplemental CSV cannot replace them.
    """
    original = pd.read_excel(workbook_path, engine="openpyxl")
    if tuple(original.columns) != WORKBOOK_COLUMNS:
        raise ValueError(f"The original workbook must have columns {WORKBOOK_COLUMNS}.")
    original_ids = _record_ids(original["NO."], "Original workbook")
    for column in ("log k", "pH"):
        measured = pd.to_numeric(original[column], errors="raise")
        if not np.isfinite(measured.to_numpy(dtype=float)).all():
            raise ValueError(
                f"Original workbook: {column} contains missing/nonfinite values."
            )

    supplied = pd.read_csv(
        inputs_path, encoding="utf-8-sig", dtype=str, keep_default_na=False
    )
    missing = set(INPUT_COLUMNS) - set(supplied.columns)
    unexpected = set(supplied.columns) - {*INPUT_COLUMNS, "CAS"}
    if missing or unexpected:
        raise ValueError(
            f"Inputs CSV columns: missing={sorted(missing)}, unexpected={sorted(unexpected)}; "
            "provide NO., SMILES, category27_label and optional CAS."
        )
    supplied_ids = _record_ids(supplied["NO."], "Inputs CSV")
    missing_ids = sorted(set(original_ids) - set(supplied_ids))
    extra_ids = sorted(set(supplied_ids) - set(original_ids))
    if missing_ids or extra_ids:
        raise ValueError(
            f"Inputs CSV identifier coverage: missing={missing_ids}, extra={extra_ids}."
        )
    supplied.index = supplied_ids
    aligned = supplied.loc[original_ids].reset_index(drop=True)
    structures = aligned["SMILES"].map(
        lambda value: "".join(value.replace("\u200b", "").replace("\ufeff", "").split())
    )
    invalid_structures = [
        int(identifier)
        for identifier, smiles in zip(original_ids, structures)
        if not smiles or Chem.MolFromSmiles(smiles) is None
    ]
    if invalid_structures:
        raise ValueError(
            f"Inputs CSV has invalid/missing SMILES for NO.: {invalid_structures}"
        )
    labels = aligned["category27_label"].str.strip()
    invalid_labels = ~labels.isin(LOCKED_CATEGORY27_OPTIONS)
    if invalid_labels.any():
        invalid_ids = original_ids.loc[invalid_labels].astype(int).tolist()
        raise ValueError(
            f"Inputs CSV has invalid/missing category27_label for NO.: {invalid_ids}"
        )
    if set(labels) != set(LOCKED_CATEGORY27_OPTIONS):
        raise ValueError(
            "Canonical training requires all 25 documented category labels."
        )

    combined = original.copy()
    combined["chemical compound"] = original["Compounds"]
    combined["logk"] = original["log k"]
    combined["SMILES"] = structures
    combined["category27_label"] = labels
    combined["category27_code"] = labels.str.split(":", n=1).str[0]
    combined["category27_name"] = labels.str.split(":", n=1).str[1].str.strip()
    if "CAS" in aligned:
        combined["CAS"] = aligned["CAS"]
    return combined


def prepare_training_inputs(
    workbook_path: Path,
    inputs_path: Path,
    output_root: Path,
    *,
    search_seeds: int = 512,
) -> tuple[Path, Path]:
    """Write model inputs and an algorithmic split into local training outputs."""
    from .splitting import build_condition_split

    dataframe = load_training_inputs(workbook_path, inputs_path)
    prepared_root = Path(output_root).resolve() / "prepared"
    prepared_root.mkdir(parents=True, exist_ok=True)
    data_path = prepared_root / "model_inputs.csv"
    split_path = prepared_root / "split.json"
    dataframe.to_csv(data_path, index=False, encoding="utf-8", lineterminator="\n")
    # Use the same CSV parser as the dataset and original split generator.
    active_dataframe = pd.read_csv(data_path, encoding="utf-8-sig", low_memory=False)
    payload = build_condition_split(active_dataframe, search_seeds=search_seeds)
    payload.update(
        source_csv=data_path.name,
        source_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        original_workbook_sha256=hashlib.sha256(
            Path(workbook_path).read_bytes()
        ).hexdigest(),
        supplied_inputs_sha256=hashlib.sha256(
            Path(inputs_path).read_bytes()
        ).hexdigest(),
    )
    split_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return data_path, split_path
