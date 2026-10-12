"""Chemical-name resolution, molecular fingerprints and predictor class assignments."""

from __future__ import annotations

import io
import os
import re
from typing import Dict, Optional, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

FP_SIZE = 3147
RADIUS = 2

LOCKED_CATEGORY27_LABELS = {
    "A": "A: Other aqueous environmental species",
    "B": "B: Alcohol",
    "C": "C: Diol",
    "D": "D: Ether",
    "E": "E: Ketone",
    "F": "F: Aldehyde",
    "G": "G: Ester",
    "H": "H: Carboxyl",
    "I": "I: Dicarboxylic",
    "J": "J: Halogenated",
    "K": "K: Sulfide, disulfide",
    "L": "L: Oxidized sulfur compounds",
    "M": "M: Thiol",
    "N": "N: Nitrile",
    "O": "O: Nitro",
    "P": "P: Amide",
    "Q": "Q: Amine",
    "R": "R: Nitroso, nitramine",
    "S": "S: Phosphorus",
    "T": "T: Cyclo",
    "U": "U: Alkene",
    "V": "V: Residual aromatic and phenolic compounds",
    "W": "W: Six-membered N heteroaromatics",
    "Y": "Y: Urea",
    "Z": "Z: Five-membered N heteroaromatics (>=2 N)",
}
LOCKED_CATEGORY27_OPTIONS = list(LOCKED_CATEGORY27_LABELS.values())


def read_csv_robust(path, **kwargs):
    """Read a CSV with common encoding fallbacks."""
    kwargs = dict(kwargs)
    kwargs.setdefault("low_memory", False)
    encodings_order = ["utf-8", "utf-8-sig", "gb18030", "gbk", "cp936", "latin1"]
    last_err = None
    for enc in encodings_order:
        try:
            df_tmp = pd.read_csv(path, encoding=enc, **kwargs)
            return df_tmp
        except UnicodeDecodeError as e:
            last_err = e
        except Exception as e:
            last_err = e

    try:
        with open(path, "rb") as f:
            raw = f.read()
        for enc in encodings_order:
            try:
                text = raw.decode(enc, errors="ignore")
                buf = io.StringIO(text)
                safe_kwargs = {k: v for k, v in kwargs.items() if k != "encoding"}
                df_tmp = pd.read_csv(buf, **safe_kwargs)
                return df_tmp
            except Exception:
                continue
    except Exception as e:
        last_err = e
    raise last_err or UnicodeDecodeError("Cannot decode the CSV file")


def get_cas_from_chemsrc(chemical_name):
    """Query Chemsrc for a CAS number."""
    if not chemical_name or chemical_name.strip() == "":
        return None
    try:
        encoded_name = quote(chemical_name.strip())
        url = f"https://www.chemsrc.com/searchResult/{encoded_name}/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        match = re.search(r"\b\d{2,7}-\d{2}-\d\b", soup.get_text())
        if match:
            return match.group()
        return None
    except Exception:
        return None


def get_cas_from_pubchem(chemical_name):
    """Extract a CAS number from PubChem synonyms."""
    if not chemical_name or str(chemical_name).strip() == "":
        return None
    try:
        encoded = quote(str(chemical_name).strip())
        cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/cids/TXT"
        cid_resp = requests.get(cid_url, timeout=12)
        if cid_resp.status_code != 200:
            return None
        cids = [
            line.strip()
            for line in cid_resp.text.splitlines()
            if line.strip().isdigit()
        ]
        if not cids:
            return None
        syn_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids[0]}/synonyms/JSON"
        syn_resp = requests.get(syn_url, timeout=12)
        if syn_resp.status_code != 200:
            return None
        data = syn_resp.json()
        info_list = data.get("InformationList", {}).get("Information", [])
        if not info_list:
            return None
        for syn in info_list[0].get("Synonym", []) or []:
            match = re.fullmatch(r"\d{2,7}-\d{2}-\d", str(syn).strip())
            if match:
                return match.group()
        return None
    except Exception:
        return None


def get_cas_number(chemical_name):
    """Query Chemsrc first, then fall back to PubChem."""
    cas = get_cas_from_chemsrc(chemical_name)
    if cas:
        return cas, "Chemsrc"
    cas = get_cas_from_pubchem(chemical_name)
    if cas:
        return cas, "PubChem"
    return None, ""


def get_smiles_via_api(compound_name, cas_number=None):
    """Resolve SMILES through PubChem or CACTUS using a name or CAS number."""
    name = (compound_name or "").strip()
    cas = (cas_number or "").strip()
    if name == "" and cas == "":
        return None

    def _canonicalize_smiles(smiles_text):
        if not smiles_text:
            return None
        try:
            mol = Chem.MolFromSmiles(str(smiles_text).strip())
            if mol is None:
                return None
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    def _read_pubchem_smiles_json(url):
        resp = requests.get(url, timeout=12)
        if resp.status_code != 200:
            return None
        data = resp.json()
        compounds = data.get("PC_Compounds", [])
        if not compounds:
            return None
        props = compounds[0].get("props", [])
        for prop in props:
            urn = prop.get("urn", {})
            if urn.get("label") == "SMILES" and urn.get("name") in (
                "Absolute",
                "Canonical",
                "Connectivity",
            ):
                val = prop.get("value", {}).get("sval")
                cano = _canonicalize_smiles(val)
                if cano:
                    return cano
        return None

    def _read_smiles_txt(url):
        resp = requests.get(url, timeout=12)
        if resp.status_code != 200:
            return None
        txt = resp.text.strip().splitlines()
        if not txt:
            return None
        return _canonicalize_smiles(txt[0].strip())

    def _query_pubchem_name(q):
        encoded = quote(q)
        urls = [
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/TXT",
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/JSON",
        ]
        for _ in range(2):
            for u in urls:
                try:
                    smi = (
                        _read_smiles_txt(u)
                        if u.endswith("/TXT")
                        else _read_pubchem_smiles_json(u)
                    )
                    if smi:
                        return smi
                except Exception:
                    continue
        return None

    def _query_pubchem_cas(cas_q):
        encoded = quote(cas_q)
        urls = [
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RN/{encoded}/property/CanonicalSMILES/TXT",
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/TXT",
        ]
        for _ in range(2):
            for u in urls:
                try:
                    smi = _read_smiles_txt(u)
                    if smi:
                        return smi
                except Exception:
                    continue
        return None

    def _query_cactus(q):
        encoded = quote(q)
        url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded}/smiles"
        for _ in range(2):
            try:
                resp = requests.get(url, timeout=12)
                if resp.status_code != 200:
                    continue
                text = resp.text.strip()
                if not text or "Page not found" in text or "Status: 404" in text:
                    continue
                smi = _canonicalize_smiles(text.splitlines()[0])
                if smi:
                    return smi
            except Exception:
                continue
        return None

    if name:
        smi = _query_pubchem_name(name)
        if smi:
            return smi
    if cas:
        smi = _query_pubchem_cas(cas)
        if smi:
            return smi
    if name:
        smi = _query_cactus(name)
        if smi:
            return smi
    if cas:
        smi = _query_cactus(cas)
        if smi:
            return smi

    return None


def smiles_to_fingerprint(smiles, fp_size=FP_SIZE):
    """Convert SMILES to a Morgan fingerprint."""
    if pd.isna(smiles) or str(smiles).strip() == "":
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None
        gen_custom = rdFingerprintGenerator.GetMorganGenerator(
            radius=RADIUS, fpSize=int(fp_size)
        )
        fp = gen_custom.GetFingerprint(mol)
        return np.array(list(fp), dtype=np.float32)
    except Exception:
        return None


def canonicalize_smiles(smiles_text: object) -> Tuple[str, str]:
    text = "" if smiles_text is None else str(smiles_text).strip()
    if not text or text.lower() == "nan":
        return "", "missing"
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return "", "invalid"
        return Chem.MolToSmiles(mol), "valid"
    except Exception:
        return "", "invalid"


def normalize_category27_input(category_input):
    """Map a user-provided hint to the locked SEEN25-1626 class taxonomy."""
    if pd.isna(category_input):
        return None
    text = str(category_input).strip()
    if text == "":
        return None
    low = text.lower()
    for code, label in LOCKED_CATEGORY27_LABELS.items():
        name = label.split(":", 1)[1].strip()
        if low in {code.lower(), label.lower(), name.lower()}:
            return locked_category_info(
                code, "Selected from the locked 25-class taxonomy.", "manual_input"
            )
    return None


def locked_category_info(
    category_label: Optional[str], reason: str = "", source: str = ""
):
    label = str(category_label or "").strip()
    if not label:
        return None
    if label.startswith("cat_"):
        label = label[4:]
    code, separator, name = label.partition(":")
    code = code.strip()
    if code not in LOCKED_CATEGORY27_LABELS:
        return None
    locked_label = LOCKED_CATEGORY27_LABELS[code]
    item = {
        "code": code,
        "name_en": locked_label.split(":", 1)[1].strip(),
        "category27_code": code,
        "category27_name": locked_label.split(":", 1)[1].strip(),
        "category27_label": locked_label,
        "label": locked_label,
    }
    if reason:
        item["category27_reason"] = str(reason)
    if source:
        item["source"] = source
    return item


def load_category_overrides(path: str) -> Dict[str, str]:
    """Load molecule-specific class assignments used by the predictor."""
    if not path or not os.path.exists(path):
        return {}
    frame = read_csv_robust(path)
    if not {"smiles", "category27_label"}.issubset(frame.columns):
        return {}
    lookup = {}
    for _, row in frame[["smiles", "category27_label"]].dropna().iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]).strip())
        if mol is None:
            continue
        category = locked_category_info(str(row["category27_label"]).strip())
        if category:
            lookup[Chem.MolToSmiles(mol, isomericSmiles=True)] = category[
                "category27_label"
            ]
    return lookup


def _has_smarts(mol, pattern: str) -> bool:
    query = Chem.MolFromSmarts(pattern)
    return bool(query is not None and mol.HasSubstructMatch(query))


def _classify_locked_category_from_mol(mol) -> Tuple[str, str]:
    """Assign one of the exact 25 model classes using the training-label contract."""
    carbon_atoms = sum(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
    if carbon_atoms == 0:
        return "A", "No carbon skeleton; assigned to aqueous environmental species."
    if _has_smarts(mol, "[PX3,PX4,PX5]"):
        return "S", "Contains a phosphorus-centered functional group."
    carboxyl_count = len(
        mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=[OX1])[O;H1,-1]"))
    )
    if carboxyl_count >= 2:
        return "I", "Contains two or more carboxyl/carboxylate groups."
    if carboxyl_count == 1:
        return "H", "Contains a carboxyl/carboxylate group."
    if _has_smarts(mol, "[NX3][CX3](=[OX1])[NX3]"):
        return "Y", "Contains a urea-type N-C(=O)-N motif."
    if _has_smarts(mol, "[S](=[OX1])(=[OX1])") or _has_smarts(mol, "[S](=[OX1])"):
        return "L", "Contains an oxidized sulfur functional group."
    if _has_smarts(mol, "[SX2H]"):
        return "M", "Contains a thiol group."
    if _has_smarts(mol, "[SX2,SX1]"):
        return "K", "Contains a sulfide or disulfide sulfur center."
    if _has_smarts(mol, "[N+](=O)[O-]"):
        return "O", "Contains a nitro group."
    if _has_smarts(mol, "[N]=O") or _has_smarts(mol, "[N][N](=O)"):
        return "R", "Contains a nitroso or nitramine motif."
    if _has_smarts(mol, "[C]#[N]"):
        return "N", "Contains a nitrile group."
    if _has_smarts(mol, "[NX3][CX3](=[OX1])"):
        return "P", "Contains an amide-type N-C(=O) motif."
    if _has_smarts(mol, "[CX3](=[OX1])[OX2]"):
        return "G", "Contains an ester-type C(=O)-O motif."
    if _has_smarts(mol, "[CX3H1](=[OX1])"):
        return "F", "Contains an aldehyde carbonyl."
    if _has_smarts(mol, "[CX3](=[OX1])([#6])[#6]"):
        return "E", "Contains a ketone carbonyl."

    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms = [mol.GetAtomWithIdx(index) for index in ring]
        nitrogen_count = sum(atom.GetAtomicNum() == 7 for atom in ring_atoms)
        if (
            len(ring) == 6
            and nitrogen_count >= 1
            and all(atom.GetIsAromatic() for atom in ring_atoms)
        ):
            return "W", "Contains a six-membered aromatic N-heterocycle."
        if (
            len(ring) == 5
            and nitrogen_count >= 2
            and all(atom.GetIsAromatic() for atom in ring_atoms)
        ):
            return (
                "Z",
                "Contains a five-membered aromatic N-heterocycle with at least two nitrogens.",
            )
    if _has_smarts(mol, "[NX3;!$(N[CX3](=O));!$(N=S)]"):
        return "Q", "Contains a non-amide amine nitrogen."
    if any(atom.GetAtomicNum() in {9, 17, 35, 53} for atom in mol.GetAtoms()):
        return "J", "Contains a carbon-bound halogen without a higher-priority group."
    if _has_smarts(mol, "[CX3;!$(C=O)]-[OX2]-[CX3;!$(C=O)]"):
        return "D", "Contains an ether-type C-O-C linkage."
    hydroxyl_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[OX2H][#6]")))
    if hydroxyl_count >= 2:
        return "C", "Contains two or more alcohol-type hydroxyl groups."
    if hydroxyl_count == 1:
        return "B", "Contains one alcohol-type hydroxyl group."
    if any(
        bond.GetBondType() == Chem.BondType.DOUBLE
        and not bond.GetIsAromatic()
        and bond.GetBeginAtom().GetAtomicNum() == 6
        and bond.GetEndAtom().GetAtomicNum() == 6
        for bond in mol.GetBonds()
    ):
        return "U", "Contains a non-aromatic carbon-carbon double bond."
    if any(atom.GetIsAromatic() for atom in mol.GetAtoms()):
        return (
            "V",
            "Residual aromatic or phenolic compound without a higher-priority group.",
        )
    if mol.GetRingInfo().NumRings() > 0:
        return "T", "Non-aromatic cyclic compound without a higher-priority group."
    return "A", "No higher-priority organic functional class was identified."


def classify_category27_from_smiles(
    smiles: Optional[str],
    manual_input: Optional[str] = None,
    category_overrides: Optional[Dict[str, str]] = None,
):
    """Assign a valid locked 25-class label from SMILES or an explicit class hint."""
    manual_info = normalize_category27_input(manual_input)
    if manual_info:
        return manual_info
    if not smiles or not str(smiles).strip():
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    override_label = (category_overrides or {}).get(canonical)
    if override_label:
        return locked_category_info(
            override_label,
            "Molecule-specific class assignment from the inference category map.",
            "category_override",
        )
    code, reason = _classify_locked_category_from_mol(mol)
    return locked_category_info(code, reason, "locked_smiles_rule")


def load_reference_dataset(dataset_path: str):
    """Load reference columns used to resolve a chemical-name input."""
    ref_path = dataset_path
    df = read_csv_robust(ref_path)
    col_lower = {col: col.strip().lower() for col in df.columns}

    def pick(*names):
        normalized_names = [
            str(name).strip().lower().replace(" ", "") for name in names
        ]
        for name in names:
            for col, low in col_lower.items():
                if (
                    low == str(name).strip().lower()
                    or low.replace(" ", "") in normalized_names
                ):
                    return col
        return None

    compound_col = pick("chemical compound", "chemical_compound", "compound")
    cas_col = pick("cas", "cas number", "CAS")
    smiles_col = pick("SMILES")
    category_col = pick("category")
    category_name_col = pick(
        "new_category_name", "family12_category_name", "category_name"
    )
    category_code_col = pick(
        "new_category_code", "family12_category_code", "category_code"
    )
    category27_scheme_col = pick("category27_scheme")
    category27_code_col = pick("category27_code")
    category27_name_col = pick("category27_name")
    category27_label_col = pick("category27_label")
    category27_reason_col = pick("category27_reason")
    if not compound_col:
        raise ValueError(
            "The reference dataset does not contain a 'chemical compound' column."
        )

    fp_cols = []
    for col in df.columns:
        if re.fullmatch(r"FP[_\s-]*\d+", str(col).strip(), flags=re.IGNORECASE):
            fp_cols.append(col)
    if fp_cols:
        fp_cols.sort(key=lambda x: int(re.findall(r"\d+", str(x))[0]))

    keep_cols = [
        c
        for c in [
            compound_col,
            cas_col,
            smiles_col,
            category_col,
            category_name_col,
            category_code_col,
            category27_scheme_col,
            category27_code_col,
            category27_name_col,
            category27_label_col,
            category27_reason_col,
        ]
        if c
    ] + fp_cols
    df = df[keep_cols]
    rename_map = {compound_col: "compound"}
    if cas_col:
        rename_map[cas_col] = "cas"
    if smiles_col:
        rename_map[smiles_col] = "smiles"
    if category_col:
        rename_map[category_col] = "category"
    if category_name_col:
        rename_map[category_name_col] = "family12_name"
    if category_code_col:
        rename_map[category_code_col] = "family12_code"
    if category27_scheme_col:
        rename_map[category27_scheme_col] = "category27_scheme"
    if category27_code_col:
        rename_map[category27_code_col] = "category27_code"
    if category27_name_col:
        rename_map[category27_name_col] = "category27_name"
    if category27_label_col:
        rename_map[category27_label_col] = "category27_label"
    if category27_reason_col:
        rename_map[category27_reason_col] = "category27_reason"
    df = df.rename(columns=rename_map)
    return df


def find_compound_record(name, df):
    """Find a reference record matching the supplied chemical name."""
    if df is None or not name:
        return None
    name_norm = str(name).strip().lower()
    df_local = df.copy()
    df_local["compound_norm"] = df_local["compound"].astype(str).str.strip().str.lower()
    hit = df_local[df_local["compound_norm"] == name_norm]
    if hit.empty:
        return None
    row = hit.iloc[0]
    category27_info = locked_category_info(
        row.get("category27_label"),
        str(row.get("category27_reason") or "Loaded from the reference dataset."),
        "reference_dataset",
    )

    fp_pairs = []
    for col in row.index:
        m = re.fullmatch(r"fp[_\s-]*(\d+)", str(col).strip().lower())
        if not m:
            continue
        try:
            fp_idx = int(m.group(1))
            fp_val = float(row.get(col))
        except Exception:
            fp_val = 0.0
            fp_idx = int(m.group(1))
        if np.isnan(fp_val):
            fp_val = 0.0
        fp_pairs.append((fp_idx, fp_val))
    fp_vec = None
    if fp_pairs:
        fp_pairs.sort(key=lambda x: x[0])
        fp_vec = np.array([v for _, v in fp_pairs], dtype=np.float32)
        if fp_vec.size <= 0:
            fp_vec = None

    return {
        "cas": row.get("cas"),
        "smiles": row.get("smiles"),
        "fingerprint": fp_vec,
        "category27_info": category27_info,
    }
