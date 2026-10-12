"""Chemical categories, functional groups and structure utilities."""

from __future__ import annotations
from typing import Dict, Optional
from rdkit import Chem
from rdkit.Chem import AllChem

CATEGORY_LABEL_PRIORITY = ("category27_label", "category27_code", "category27_name")
GLOBAL_DESCRIPTOR_COLUMNS = (
    "desc_mol_wt",
    "desc_log_p",
    "desc_tpsa",
    "desc_hbd",
    "desc_hba",
    "desc_rotatable",
    "desc_ring_count",
    "desc_fraction_csp3",
    "desc_heavy_atoms",
    "desc_formal_charge",
    "desc_aromatic_fraction",
    "desc_hetero_fraction",
)
REACTIVITY_FAMILY_BY_CATEGORY_CODE = {
    "A": "saturated_hydrocarbon",
    "T": "saturated_hydrocarbon",
    "U": "unsaturated_hydrocarbon",
    "V": "unsaturated_hydrocarbon",
    "B": "hydroxyl",
    "C": "hydroxyl",
    "D": "ether",
    "E": "neutral_carbonyl",
    "F": "neutral_carbonyl",
    "G": "carboxyl_derivative",
    "H": "carboxyl_derivative",
    "I": "carboxyl_derivative",
    "J": "halogenated",
    "K": "reduced_sulfur",
    "M": "reduced_sulfur",
    "L": "oxidized_sulfur",
    "N": "neutral_nitrogen",
    "O": "neutral_nitrogen",
    "P": "neutral_nitrogen",
    "R": "neutral_nitrogen",
    "Y": "neutral_nitrogen",
    "Q": "basic_nitrogen",
    "W": "basic_nitrogen",
    "Z": "basic_nitrogen",
    "S": "phosphorus",
}


def _clean_smiles_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    return (
        text.replace("\u200b", "")
        .replace("\ufeff", "")
        .replace("\u200e", "")
        .replace("\xa0", " ")
        .strip()
    )


class _FGDef:

    def __init__(
        self,
        name: str,
        smarts: str,
        display: str,
        highlight_idx: list,
        color=(0.95, 0.8, 0.85),
    ):
        self.name = name
        self.smarts = smarts
        self.display = display
        self.highlight_idx = list(highlight_idx)
        self.color = color
        self._pat = Chem.MolFromSmarts(smarts)

    @property
    def pattern(self):
        return self._pat


FG_DEFINITIONS = [
    _FGDef("Carboxylic acid", "[CX3](=O)[OX2H1]", "-COOH", [2], (0.98, 0.9, 0.8)),
    _FGDef("Sulfonic acid", "[SX4](=O)(=O)[OX2H1]", "-SO3H", [3], (0.98, 0.9, 0.95)),
    _FGDef(
        "Phosphonic acid",
        "[PX4](=O)([OX2H1,O-])[OX2H1,O-]",
        "-PO(OH)2",
        [1],
        (0.98, 0.85, 0.35),
    ),
    _FGDef("Ester", "[CX3](=O)[OX2][CX4]", "-COOR", [2], (0.95, 0.92, 0.8)),
    _FGDef("Amide", "[NX3][CX3](=O)[#6]", "-CONH-", [0, 1], (0.92, 0.92, 0.98)),
    _FGDef("Phenol", "c[OX2H]", "Ar-OH", [1], (0.85, 0.95, 0.85)),
    _FGDef("Alcohol", "[OX2H][CX4]", "-OH", [0], (0.85, 0.95, 0.95)),
    _FGDef("Nitro", "[N+](=O)[O-]", "-NO2", [0], (0.98, 0.9, 0.9)),
    _FGDef("Ketone", "[CX3](=O)[#6]", "-CO-", [0], (0.92, 0.92, 0.92)),
    _FGDef("Primary amine", "[NX3;H2;!$(NC=O)]", "-NH2", [0], (0.95, 0.9, 0.7)),
    _FGDef("Secondary amine", "[NX3;H1;!$(NC=O)]", "-NHR", [0], (0.9, 0.95, 0.8)),
    _FGDef("Tertiary amine", "[NX3;H0;!$(NC=O)]", "-NR2", [0], (0.85, 0.9, 0.98)),
    _FGDef("Ether", "[OD2]([#6])[#6]", "-O-", [0], (0.9, 0.95, 0.95)),
    _FGDef("Halogen", "[F,Cl,Br,I][#6]", "-X", [0], (0.9, 0.95, 0.9)),
    _FGDef("Thiol", "[SH][CX4]", "-SH", [0], (0.95, 0.95, 0.85)),
    _FGDef("Carboxylate", "[CX3](=O)[O-]", "-COO−", [2], (0.95, 0.9, 0.85)),
]
FG_REPRESENTATIVE_SMILES = {
    "Alcohol": "CCO",
    "Phenol": "c1ccccc(O)c1",
    "Ether": "COC",
    "Carboxylic acid": "CC(=O)O",
    "Carboxylate": "CC(=O)[O-]",
    "Ester": "CC(=O)OC",
    "Amide": "CC(=O)NC",
    "Ketone": "CC(=O)C",
    "Primary amine": "CN",
    "Secondary amine": "CNC",
    "Tertiary amine": "CN(C)C",
    "Nitro": "c1ccccc([N+](=O)[O-])c1",
    "Halogen": "CCCl",
    "Thiol": "CS",
    "Sulfonic acid": "CS(=O)(=O)O",
    "Phosphonic acid": "CP(=O)(O)O",
}
IONIZABLE_PHYSICS_RULES = [
    {
        "name": "Carboxylic acid",
        "smarts": "[CX3](=O)[OX2H1,O-]",
        "pka": 4.5,
        "sign": 1.0,
        "slope": 0.1,
    },
    {
        "name": "Sulfonic acid",
        "smarts": "[SX4](=O)(=O)[OX2H1,O-]",
        "pka": 1.2,
        "sign": 1.0,
        "slope": 0.06,
    },
    {
        "name": "Phosphonic acid",
        "smarts": "[PX4](=O)([OX2H1,O-])[OX2H1,O-]",
        "pka": 2.1,
        "sign": 1.0,
        "slope": 0.07,
    },
    {"name": "Phenol", "smarts": "c[OX2H,O-]", "pka": 9.8, "sign": 1.0, "slope": 0.08},
    {
        "name": "Thiol",
        "smarts": "[SH,S-][CX4,c]",
        "pka": 9.5,
        "sign": 1.0,
        "slope": 0.06,
    },
    {
        "name": "Amine",
        "smarts": "[NX3;!$(NC=O)]",
        "pka": 9.7,
        "sign": 1.0,
        "slope": 0.08,
    },
    {
        "name": "Imidazole",
        "smarts": "n1cc[nH]c1",
        "pka": 7.0,
        "sign": 1.0,
        "slope": 0.07,
    },
    {"name": "Pyridine", "smarts": "n1ccccc1", "pka": 5.2, "sign": 1.0, "slope": 0.06},
]
for _rule in IONIZABLE_PHYSICS_RULES:
    try:
        _rule["pattern"] = Chem.MolFromSmarts(str(_rule["smarts"]))
    except Exception:
        _rule["pattern"] = None


def _ionizable_physics_hits(smiles: str) -> list:
    """Return ionizable functional-group rules present in a molecule."""
    mol = None
    try:
        mol = Chem.MolFromSmiles(_clean_smiles_text(smiles))
    except Exception:
        mol = None
    if mol is None:
        return []
    hits = []
    for rule in IONIZABLE_PHYSICS_RULES:
        pat = rule.get("pattern")
        if pat is None:
            continue
        try:
            if mol.HasSubstructMatch(pat):
                hits.append(rule)
        except Exception:
            continue
    return hits


CHEMICAL_CATEGORY_NAME_MAP = {
    "0": "Alkane",
    "1": "Alcohol",
    "2": "Diol",
    "3": "Ether",
    "4": "Ketone",
    "5": "Aldehyde",
    "6": "Ester",
    "7": "Carboxyl",
    "8": "Dicarboxylic",
    "9": "Halogenated",
    "10": "Sulfide / Disulfide",
    "11": "Sulfoxide",
    "12": "Thiol",
    "13": "Nitrile",
    "14": "Nitro",
    "15": "Amide",
    "16": "Amine",
    "17": "Nitroso / Nitramine",
    "18": "Phosphorus",
    "19": "Cyclo",
    "20": "Alkene",
    "21": "Benzene",
    "22": "Pyridine",
    "23": "Furan",
    "24": "Urea",
    "25": "Imidazole",
    "26": "Triazine",
}
TRIVIAL_SUBSTRUCTURE_EXACT = {
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "CO",
    "CN",
    "CCN",
    "CC",
    "CCC",
    "cO",
    "cN",
    "cS",
    "C=CC",
    "C:C:N",
}


def _env_smiles_from_center(mol: Chem.Mol, center: int, rad: int) -> str:
    try:
        bonds = AllChem.FindAtomEnvironmentOfRadiusN(mol, int(rad), int(center))
        if not bonds:
            atom = mol.GetAtomWithIdx(int(center))
            em = Chem.EditableMol(Chem.Mol())
            _ = em.AddAtom(Chem.Atom(atom.GetAtomicNum()))
            submol = em.GetMol()
        else:
            submol = Chem.PathToSubmol(mol, bonds)
        return Chem.MolToSmiles(submol, canonical=True)
    except Exception:
        return ""


def _canonicalize_smiles_or_smarts(smi: str) -> str:
    if not isinstance(smi, str) or not smi:
        return smi
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmarts(smi)
        if mol is None:
            return smi
        return Chem.MolToSmiles(mol)
    except Exception:
        return smi


def _is_informative_substructure(smi: str, min_atoms: int = 2) -> bool:
    if not isinstance(smi, str) or not smi or str(smi).startswith("Unknown"):
        return False
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmarts(smi)
        if mol is None:
            return False
        return int(mol.GetNumAtoms()) >= int(min_atoms)
    except Exception:
        return False


def _detect_functional_group(sub_smiles: str) -> Optional[Dict[str, object]]:
    if not sub_smiles or str(sub_smiles).startswith("Unknown"):
        return None
    mol = None
    try:
        mol = Chem.MolFromSmiles(sub_smiles)
    except Exception:
        mol = None
    if mol is None:
        try:
            mol = Chem.MolFromSmarts(sub_smiles)
        except Exception:
            mol = None
    if mol is None:
        return None
    for fg in FG_DEFINITIONS:
        pat = fg.pattern
        if pat is None:
            continue
        try:
            if mol.HasSubstructMatch(pat):
                match = mol.GetSubstructMatches(pat, uniquify=True)
                if not match:
                    continue
                first = match[0]
                highlight_atoms = (
                    [first[i] for i in fg.highlight_idx if i < len(first)]
                    if fg.highlight_idx
                    else list(first)
                )
                return {
                    "fg_name": fg.name,
                    "display": fg.display,
                    "highlight_atoms": highlight_atoms,
                    "color": fg.color,
                    "smarts": fg.smarts,
                }
        except Exception:
            continue
    return None


def _functional_group_def_by_name(name: str) -> Optional[_FGDef]:
    target = str(name).strip()
    if not target:
        return None
    for fg in FG_DEFINITIONS:
        if str(fg.name) == target:
            return fg
    return None


def _pretty_category_label(label: object) -> str:
    raw = str(label).strip()
    if raw.startswith("cat_"):
        raw = raw.replace("cat_", "", 1)
    mapped = CHEMICAL_CATEGORY_NAME_MAP.get(raw, raw)
    mapped = str(mapped).replace("_", " ").strip()
    return mapped
