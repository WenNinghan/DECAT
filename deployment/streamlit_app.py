"""Streamlit interface for aqueous ozone reaction-rate prediction with DECAT."""

import os
import re
import io
import base64
import json
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from html import escape

import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from streamlit.runtime.scriptrunner import get_script_run_ctx

SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = SCRIPT_DIR.parent
RUNTIME_PACKAGE = RELEASE_ROOT / "core" / "locked_package"
RUNTIME_CONFIG_PATH = RUNTIME_PACKAGE / "configs" / "decat.json"
RUNTIME_REFERENCE_PATH = RUNTIME_PACKAGE / "configs" / "preprocessing.json"
DEPLOYMENT_ARTIFACTS = SCRIPT_DIR / "artifacts"
CATEGORY_OVERRIDES_PATH = SCRIPT_DIR / "data" / "category_overrides.csv"
PACKAGE_SOURCE = RUNTIME_PACKAGE / "src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from decat import inference, input_chemistry
from decat.inference import predict, predict_ensemble
from decat.input_chemistry import (
    FP_SIZE,
    LOCKED_CATEGORY27_OPTIONS,
    canonicalize_smiles,
    find_compound_record,
    normalize_category27_input,
)

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.info")


@st.cache_resource
def load_predictor():
    return inference.load_predictor(RUNTIME_PACKAGE, DEPLOYMENT_ARTIFACTS)


@st.cache_data(ttl=3600)
def get_cas_number(chemical_name):
    return input_chemistry.get_cas_number(chemical_name)


@st.cache_data(ttl=3600)
def get_smiles_via_api(compound_name, cas_number=None):
    smiles = input_chemistry.get_smiles_via_api(compound_name, cas_number)
    if smiles is None:
        st.warning(
            "SMILES lookup failed: neither PubChem nor CACTUS returned a valid structure. Please check the network, DNS, or compound name."
        )
    return smiles


def smiles_to_fingerprint(smiles, fp_size=FP_SIZE):
    fingerprint = input_chemistry.smiles_to_fingerprint(smiles, fp_size)
    if fingerprint is None:
        st.warning("Fingerprint generation failed for the supplied SMILES.")
    return fingerprint


@st.cache_data(show_spinner=False)
def load_category_overrides(path: str):
    return input_chemistry.load_category_overrides(path)


def classify_category27_from_smiles(smiles, manual_input=None):
    return input_chemistry.classify_category27_from_smiles(
        smiles,
        manual_input=manual_input,
        category_overrides=load_category_overrides(str(CATEGORY_OVERRIDES_PATH)),
    )


@st.cache_data(ttl=3600)
def load_reference_dataset(config_path=None):
    try:
        config = json.loads(
            Path(config_path or RUNTIME_REFERENCE_PATH).read_text(encoding="utf-8")
        )
        return pd.DataFrame(config["compound_lookup"])
    except Exception as error:
        st.warning(f"Reference dataset loading failed: {error}")
        return None


def inject_custom_styles():
    st.markdown(
        """
        <style>
            :root {
                --bg: #f5f7fa;
                --bg-2: #ffffff;
                --panel: #ffffff;
                --panel-strong: #ffffff;
                --border: rgba(0, 51, 102, 0.2);
                --text: #000000;
                --muted: #555555;
                --cyan: #0066cc;
                --teal: #0066cc;
                --gold: #ff8c00;
                --gold-soft: rgba(255, 140, 0, 0.1);
                --success: #228b22;
                --warning: #ff8c00;
                --danger: #dc143c;
                --info: #0066cc;
                --shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                --radius-xl: 8px;
                --radius-lg: 6px;
                --radius-md: 4px;
            }

            .stApp {
                background: linear-gradient(180deg, #f5f7fa 0%, #ffffff 100%);
                color: var(--text);
                font-family: "Times New Roman", Times, serif;
            }

            [data-testid="stAppViewContainer"] > .main {
                background: #ffffff;
                padding: 0;
            }

            [data-testid="stMarkdownContainer"] {
                font-family: "Times New Roman", Times, serif !important;
                font-size: 16pt !important;
            }

            [data-testid="stMarkdownContainer"] h3 {
                font-family: "Times New Roman", Times, serif !important;
                font-size: 24pt !important;
                font-weight: bold !important;
                color: #003366 !important;
                margin-bottom: 1.5rem !important;
                padding-bottom: 0.8rem !important;
                border-bottom: 4px solid #0066cc !important;
            }

            [data-testid="stMarkdownContainer"] strong {
                font-family: "Times New Roman", Times, serif !important;
                font-size: 16pt !important;
                font-weight: bold !important;
            }

            [data-testid="stMarkdownContainer"] p {
                font-family: "Times New Roman", Times, serif !important;
                font-size: 16pt !important;
                margin: 0.5rem 0 !important;
                line-height: 1.6 !important;
            }

            [data-testid="stHeader"] {
                background: transparent;
                display: none;
            }

            [data-testid="stToolbar"] {
                display: none;
            }

            #MainMenu, footer {
                visibility: hidden;
            }

            .block-container {
                padding: 1.5rem 3rem;
                max-width: 100%;
            }

            [data-testid="stSidebar"] {
                display: none;
            }

            .hero-shell {
                background: linear-gradient(135deg, #003366 0%, #004d99 100%);
                color: #ffffff;
                padding: 2rem 2rem;
                margin: -1.5rem -2rem 2rem -2rem;
                border-radius: 0;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            }

            .hero-title {
                margin: 0;
                font-size: 32pt;
                font-weight: bold;
                color: #ffffff;
                font-family: "Times New Roman", Times, serif;
                text-align: center;
                letter-spacing: 0.5px;
            }

            .hero-subtitle {
                display: none;
            }

            .section-label {
                font-size: 16pt;
                font-weight: bold;
                color: var(--text);
                margin: 1rem 0 0.5rem 0;
                font-family: "Times New Roman", Times, serif;
            }

            .grid-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.5rem;
                margin-bottom: 1.5rem;
            }

            .section-box {
                background: var(--panel);
                border: 2px solid var(--border);
                border-radius: 4px;
                padding: 1.2rem;
            }

            .section-title {
                font-size: 14pt;
                font-weight: bold;
                color: var(--text);
                margin: 0 0 0.8rem 0;
                font-family: "Times New Roman", Times, serif;
            }

            .input-group {
                margin-bottom: 0.8rem;
            }

            .input-label {
                font-size: 11pt;
                font-weight: bold;
                color: var(--text);
                margin-bottom: 0.3rem;
                display: block;
                font-family: "Times New Roman", Times, serif;
            }

            .result-row {
                display: flex;
                justify-content: space-between;
                padding: 0.4rem 0;
                border-bottom: 1px solid var(--border);
                font-size: 11pt;
                font-family: "Times New Roman", Times, serif;
            }

            .result-row:last-child {
                border-bottom: none;
            }

            .result-key {
                font-weight: bold;
                color: var(--text);
            }

            .result-value {
                color: var(--muted);
                text-align: right;
                word-break: break-word;
                max-width: 60%;
            }

            .prediction-box {
                background: #e8f4f8;
                border: 3px solid var(--teal);
                border-radius: 4px;
                padding: 1.5rem;
                text-align: center;
            }

            .prediction-label {
                font-size: 12pt;
                font-weight: bold;
                color: var(--text);
                margin-bottom: 0.5rem;
                font-family: "Times New Roman", Times, serif;
            }

            .prediction-value {
                font-size: 36pt;
                font-weight: bold;
                color: var(--teal);
                margin: 0.3rem 0;
                font-family: "Times New Roman", Times, serif;
            }

            .prediction-unit {
                font-size: 12pt;
                color: var(--muted);
                font-family: "Times New Roman", Times, serif;
            }

            [data-testid="stAppViewContainer"] > .main {
                background: transparent;
            }

            [data-testid="stHeader"] {
                background: transparent;
                display: none;
            }

            [data-testid="stToolbar"] {
                display: none;
            }

            #MainMenu, footer {
                visibility: hidden;
            }

            .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
                max-width: 1200px;
            }

            [data-testid="stSidebar"] {
                display: none;
            }

            .hero-shell {
                position: relative;
                overflow: hidden;
                padding: 1.1rem 1.5rem;
                border-radius: 6px;
                border: 2px solid var(--border);
                background: #ffffff;
                box-shadow: none;
                margin-bottom: 1.15rem;
            }

            .hero-shell::before,
            .hero-shell::after {
                display: none;
            }

            @keyframes floatOrb {
                0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
                50% { transform: translate3d(10px, -12px, 0) scale(1.06); }
            }

            .hero-kicker {
                display: none;
            }

            .hero-title {
                margin: 0;
                font-size: 24pt;
                line-height: 1.2;
                letter-spacing: -0.01em;
                color: var(--text);
                font-family: "Times New Roman", Times, serif;
                font-weight: bold;
                text-align: center;
            }

            .hero-subtitle {
                display: block;
                margin-top: 0.55rem;
                color: var(--muted);
                font-size: 12.5pt;
                line-height: 1.4;
                text-align: center;
                font-family: "Times New Roman", Times, serif;
            }

            .hero-status {
                margin-top: 0.6rem;
                text-align: center;
                font-size: 11pt;
                color: var(--teal);
                font-weight: bold;
                font-family: "Times New Roman", Times, serif;
            }

            .badge-row {
                display: none;
            }

            .status-pill {
                display: none;
            }

            .status-pill strong {
                display: none;
            }

            .insight-card,
            .control-card,
            .step-card,
            .result-card,
            .taxonomy-card,
            .feature-card,
            .alert-banner,
            .workflow-card {
                display: none;
            }

            .detail-card {
                position: relative;
                overflow: hidden;
                border-radius: 4px;
                border: 2px solid var(--border);
                background: var(--panel);
                box-shadow: none;
            }

            .feature-card {
                padding: 1.5rem;
                min-height: 100px;
                transition: none;
            }

            .feature-card:hover,
            .insight-card:hover,
            .result-card:hover,
            .taxonomy-card:hover {
                transform: none;
                border-color: var(--border);
                box-shadow: none;
            }

            .feature-icon,
            .insight-label,
            .result-label,
            .step-index {
                font-size: 11pt;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                color: var(--teal);
                font-weight: bold;
                font-family: "Times New Roman", Times, serif;
            }

            .feature-title,
            .insight-title,
            .section-title,
            .result-value {
                color: var(--text);
                font-family: "Times New Roman", Times, serif;
            }

            .feature-title {
                font-size: 16pt;
                font-weight: bold;
                margin: 0.6rem 0 0.4rem;
            }

            .feature-desc,
            .insight-desc,
            .section-desc,
            .step-message,
            .taxonomy-desc,
            .result-meta {
                color: var(--muted);
                line-height: 1.5;
                font-size: 11pt;
                font-family: "Times New Roman", Times, serif;
            }

            .section-shell {
                margin: 1.2rem 0 0.8rem;
            }

            .section-kicker {
                display: none;
            }

            .section-title {
                font-size: 20pt;
                margin: 0 0 1rem 0;
                letter-spacing: 0;
                font-weight: bold;
                font-family: "Times New Roman", Times, serif;
            }

            .section-desc {
                display: none;
            }

            .control-card,
            .insight-card,
            .result-card,
            .taxonomy-card {
                display: none;
            }

            .insight-value {
                font-size: 18pt;
                font-weight: bold;
                color: var(--text);
                margin: 0.4rem 0;
                font-family: "Times New Roman", Times, serif;
            }

            .step-card {
                padding: 1.2rem;
                margin-bottom: 0.8rem;
            }

            .step-card.success { border-color: var(--success); background: #ffffff; }
            .step-card.warn { border-color: var(--warning); background: #ffffff; }
            .step-card.error { border-color: var(--danger); background: #ffffff; }
            .step-card.info { border-color: var(--info); background: #ffffff; }

            .step-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 1rem;
                margin-bottom: 0.4rem;
            }

            .step-title {
                font-size: 14pt;
                font-weight: bold;
                color: var(--text);
                font-family: "Times New Roman", Times, serif;
            }

            .step-state {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 0.4rem 0.8rem;
                border-radius: 4px;
                font-size: 10pt;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-family: "Times New Roman", Times, serif;
            }

            .step-state.success { background: var(--success); color: #ffffff; }
            .step-state.warn { background: var(--warning); color: #ffffff; }
            .step-state.error { background: var(--danger); color: #ffffff; }
            .step-state.info { background: var(--info); color: #ffffff; }

            .result-card.main-result {
                display: block !important;
                padding: 2.5rem;
                background: linear-gradient(135deg, #e8f4f8 0%, #d4ebf7 100%);
                border: 4px solid var(--teal);
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 8px 24px rgba(0, 102, 204, 0.2);
                margin-top: 1.5rem;
            }

            .result-value {
                font-size: 56pt;
                font-weight: bold;
                line-height: 1.1;
                margin: 0.5rem 0;
                letter-spacing: 0;
                color: var(--teal);
                font-family: "Times New Roman", Times, serif;
                text-align: center;
                text-shadow: 2px 2px 4px rgba(0, 102, 204, 0.1);
            }

            .result-unit {
                color: var(--muted);
                font-size: 16pt;
                letter-spacing: 0;
                font-weight: normal;
                font-family: "Times New Roman", Times, serif;
            }

            .result-label {
                font-size: 18pt;
                font-weight: bold;
                color: #003366;
                margin-bottom: 1.15rem;
                font-family: "Times New Roman", Times, serif;
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            .result-grid-note {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-top: 1.2rem;
            }

            .workflow-shell {
                margin: 0.8rem 0 1rem;
            }

            .workflow-card {
                min-height: 146px;
                padding: 0.9rem 0.95rem;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }

            .workflow-step {
                font-size: 10.5pt;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-weight: bold;
                font-family: "Times New Roman", Times, serif;
            }

            .workflow-title {
                margin-top: 0.35rem;
                font-size: 13pt;
                line-height: 1.25;
                color: var(--text);
                font-weight: bold;
                font-family: "Times New Roman", Times, serif;
            }

            .workflow-detail {
                margin-top: 0.7rem;
                font-size: 10.5pt;
                line-height: 1.35;
                color: var(--muted);
                font-family: "Times New Roman", Times, serif;
            }

            .workflow-state {
                align-self: flex-start;
                margin-top: 0.85rem;
                padding: 0.32rem 0.65rem;
                border-radius: 999px;
                font-size: 9.5pt;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #ffffff;
                font-family: "Times New Roman", Times, serif;
            }

            .workflow-card.pending .workflow-state { background: #7a7a7a; }
            .workflow-card.success .workflow-state { background: var(--success); }
            .workflow-card.warn .workflow-state { background: var(--warning); }
            .workflow-card.error .workflow-state { background: var(--danger); }
            .workflow-card.info .workflow-state { background: var(--info); }

            .workflow-card.pending { border-color: rgba(0, 0, 0, 0.18); }
            .workflow-card.success { border-color: var(--success); }
            .workflow-card.warn { border-color: var(--warning); }
            .workflow-card.error { border-color: var(--danger); }
            .workflow-card.info { border-color: var(--info); }

            .detail-card {
                margin-bottom: 1.15rem;
                background: var(--panel);
                border: 2px solid var(--border);
                border-radius: 8px;
                padding: 1.5rem;
                box-shadow: var(--shadow);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }

            .detail-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 16px rgba(0, 102, 204, 0.15);
            }

            .detail-title {
                margin: 0 0 1rem 0;
                font-size: 18pt;
                line-height: 1.2;
                font-weight: bold;
                color: var(--text);
                font-family: "Times New Roman", Times, serif;
            }

            .detail-grid {
                display: grid;
                gap: 0.6rem;
            }

            .detail-row {
                display: grid;
                grid-template-columns: 140px 1fr;
                gap: 0.8rem;
                align-items: start;
                padding: 0.4rem 0;
                border-bottom: 1px solid var(--border);
            }

            .detail-row:last-child {
                border-bottom: none;
            }

            .detail-key {
                font-size: 14pt;
                line-height: 1.4;
                color: var(--text);
                font-weight: bold;
                font-family: "Times New Roman", Times, serif;
            }

            .detail-value {
                font-size: 14pt;
                line-height: 1.4;
                color: var(--muted);
                word-break: break-word;
                font-family: "Times New Roman", Times, serif;
            }

            .input-note {
                margin: 0.35rem 0 0.2rem;
                font-size: 10.5pt;
                line-height: 1.35;
                color: var(--muted);
                font-family: "Times New Roman", Times, serif;
            }

            .taxonomy-code {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.5rem 1rem;
                border-radius: 4px;
                background: var(--gold-soft);
                color: var(--gold);
                font-weight: bold;
                margin-bottom: 1.15rem;
                font-size: 12pt;
                font-family: "Times New Roman", Times, serif;
            }

            .alert-banner {
                padding: 0.9rem 1rem;
                margin-top: 0.7rem;
            }

            .alert-banner.error {
                border-color: var(--danger);
                background: #ffffff;
            }

            .alert-banner.success {
                border-color: var(--success);
                background: #ffffff;
            }

            [data-testid="stForm"],
            [data-testid="stVerticalBlockBorderWrapper"]:has(.control-card) {
                background: transparent !important;
                border: none !important;
            }

            .stTextInput > div > div,
            .stNumberInput > div > div,
            .stSelectbox > div > div,
            .stTextArea textarea {
                background: #ffffff !important;
                border: 2px solid var(--border) !important;
                border-radius: 6px !important;
                color: var(--text) !important;
                font-family: "Times New Roman", Times, serif !important;
                font-size: 16pt !important;
                padding: 0.8rem !important;
            }

            .stTextInput input,
            .stNumberInput input {
                font-size: 17pt !important;
                font-weight: bold !important;
                line-height: 1.1 !important;
                height: auto !important;
                min-height: 0 !important;
                padding-top: 0 !important;
                padding-bottom: 0 !important;
                font-family: "Times New Roman", Times, serif !important;
            }

            .stTextInput [data-baseweb="input"],
            .stNumberInput [data-baseweb="input"] {
                min-height: 3.05rem !important;
                height: 3.05rem !important;
                display: flex !important;
                align-items: center !important;
                overflow: visible !important;
            }

            .stTextInput label,
            .stNumberInput label,
            .stFileUploader label,
            .stRadio label,
            .stSelectbox label {
                color: var(--text) !important;
                font-weight: bold !important;
                font-size: 18pt !important;
                font-family: "Times New Roman", Times, serif !important;
            }

            .stButton > button {
                min-height: 4rem;
                border: none;
                border-radius: 8px;
                background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
                color: #ffffff;
                font-weight: bold;
                letter-spacing: 0.5px;
                font-size: 18pt;
                box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
                transition: all 0.3s ease;
                font-family: "Times New Roman", Times, serif;
                width: 100%;
            }

            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0, 102, 204, 0.4);
                background: linear-gradient(135deg, #0052a3 0%, #003d7a 100%);
            }

            [data-testid="stForm"],
            [data-testid="stVerticalBlockBorderWrapper"] {
                background: transparent !important;
                border: none !important;
            }

            div[data-testid="stDataFrame"] {
                border-radius: 4px;
                overflow: hidden;
                border: 1px solid var(--border);
                font-family: "Times New Roman", Times, serif;
            }

            [data-testid="stMetric"] {
                background: var(--panel);
                border: 1px solid var(--border);
                padding: 0.8rem;
                border-radius: 4px;
            }

            [data-testid="stMetricLabel"] {
                font-size: 10pt !important;
                font-weight: bold !important;
                color: var(--muted) !important;
                font-family: "Times New Roman", Times, serif !important;
            }

            [data-testid="stMetricValue"] {
                font-size: 16pt !important;
                font-weight: bold !important;
                color: var(--text) !important;
                font-family: "Times New Roman", Times, serif !important;
            }

            .stTabs [data-baseweb="tab-list"] {
                display: none;
            }

            .stCaptionContainer, .stCaption {
                font-size: 9pt !important;
                color: var(--muted) !important;
                font-family: "Times New Roman", Times, serif !important;
            }

            .small-caption {
                font-size: 9pt;
                color: var(--muted);
                font-family: "Times New Roman", Times, serif;
            }

            @media (max-width: 900px) {
                .grid-container {
                    grid-template-columns: 1fr;
                }
            }

            @media (max-width: 900px) {
                .hero-shell {
                    padding: 1rem;
                }
                .section-title {
                    font-size: 1.42rem;
                }
                .workflow-card {
                    min-height: 128px;
                }
                .detail-row {
                    grid-template-columns: 1fr;
                    gap: 0.15rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(dataset_ready: bool):
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #003366 0%, #004d99 100%);
                    color: white;
                    padding: 1.5rem 2rem;
                    margin: -1.5rem -2rem 2rem -2rem;
                    text-align: center;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
            <h1 style="font-size: 28pt;
                       font-weight: bold;
                       margin: 0;
                       font-family: 'Times New Roman', Times, serif;
                       letter-spacing: 1px;
                       white-space: nowrap;">
                Ozone Reaction Rate Constant (logk) Prediction System
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_board(step_events: List[Dict[str, str]]):
    base_steps = [
        (1, "Compound Query", "Input compound name and pH."),
        (2, "CAS Retrieval", "Resolve the registry identifier."),
        (3, "Structure Retrieval", "Obtain the canonical SMILES string."),
        (4, "Class Assignment", "Assign the reaction class automatically."),
        (5, "Fingerprint Encoding", "Build the Morgan fingerprint."),
        (6, "logk Prediction", "Return the final rate constant."),
        (
            7,
            "Prediction Range",
            "Assess the compound's applicability domain against the training data.",
        ),
    ]
    event_map = {
        int(item.get("index", 0)): item for item in step_events if item.get("index")
    }
    cols = st.columns(len(base_steps), gap="small")
    for col, (index, title, default_detail) in zip(cols, base_steps):
        item = event_map.get(index, {})
        tone = str(item.get("tone", "pending")).lower()
        if tone == "warning":
            tone = "warn"
        if tone not in {"pending", "success", "warn", "error", "info"}:
            tone = "info" if item else "pending"
        state = str(item.get("state", "Pending" if not item else "Ready"))
        detail = re.sub(r"\s+", " ", str(item.get("message") or default_detail)).strip()
        if len(detail) > 86:
            detail = detail[:83].rstrip() + "..."
        with col:
            st.markdown(
                f"""
                <div class="workflow-card {escape(tone)}">
                    <div>
                        <div class="workflow-step">Step {index}</div>
                        <div class="workflow-title">{escape(title)}</div>
                        <div class="workflow-detail">{escape(detail)}</div>
                    </div>
                    <div class="workflow-state">{escape(state)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_detail_card(title: str, rows: List[Tuple[str, str]]):
    st.markdown(
        f"""
        <div style="background: white;
                    border: 3px solid #003366;
                    border-radius: 8px;
                    padding: 1.5rem;
                    margin-bottom: 1.5rem;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <h3 style="font-size: 22pt;
                       font-weight: bold;
                       color: #003366;
                       margin: 0 0 1rem 0;
                       padding-bottom: 0.5rem;
                       border-bottom: 3px solid #0066cc;
                       font-family: 'Times New Roman', Times, serif;">
                {title}
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for key, value in rows:
        display_value = (
            "Not available" if value is None or str(value).strip() == "" else str(value)
        )
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(
                f"<p style='font-size: 16pt; font-weight: bold; margin: 0.5rem 0; font-family: Times New Roman;'>{key}:</p>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"<p style='font-size: 16pt; margin: 0.5rem 0; font-family: Times New Roman;'>{display_value}</p>",
                unsafe_allow_html=True,
            )


def render_alert_banner(message: str, tone: str = "error"):
    st.markdown(
        f"""
        <div class="alert-banner {escape(tone)}">
            <div class="step-title">{escape(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_range_card(detail: Dict[str, object]):
    ad_label = str(detail.get("DECAT_AD_label", "Not assessed"))
    ad_color = "#dc3545"
    st.markdown(
        f"""
        <div style="background: #fff5f5;
                    border-left: 4px solid {ad_color};
                    padding: 1rem 1rem;
                    margin-top: 0.8rem;
                    margin-bottom: 0.8rem;
                    min-height: 160px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;">
            <div style="font-size: 17pt;
                        font-weight: 800;
                        color: #003366;
                        line-height: 1.2;
                        margin-bottom: 0.5rem;
                        font-family: 'Times New Roman', Times, serif;">
                Prediction Applicability Domain
            </div>
            <div style="font-size: 18pt;
                        font-weight: 800;
                        color: {ad_color};
                        font-family: 'Times New Roman', Times, serif;">
                {escape(ad_label)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def category27_reason_for_display(info: Optional[Dict[str, str]]) -> str:
    if not info:
        return "Not available."
    return str(
        info.get("category27_reason")
        or info.get("reason")
        or info.get("reason_raw")
        or "Category27 assignment completed."
    )


def category27_label_for_display(info: Optional[Dict[str, str]]) -> str:
    if not info:
        return "Not assigned"
    label = str(info.get("category27_label") or info.get("label") or "").strip()
    if label:
        return label
    code = str(info.get("category27_code") or info.get("code") or "").strip()
    name = str(info.get("category27_name") or info.get("name_en") or "").strip()
    return f"{code}: {name}".strip(": ").strip() if (code or name) else "Not assigned"


def main():
    st.set_page_config(
        page_title="Reaction Rate Constant Prediction Workflow",
        page_icon="📄",
        layout="wide",
    )
    inject_custom_styles()

    reference_dataset_path = str(RUNTIME_REFERENCE_PATH)

    render_hero(dataset_ready=os.path.exists(reference_dataset_path))
    if not os.path.exists(reference_dataset_path):
        render_alert_banner(
            "The model runtime configuration is unavailable. Restore it before running a prediction.",
            tone="error",
        )

    main_cols = st.columns([1.05, 1.3, 1.25], gap="large", vertical_alignment="top")

    with main_cols[0]:
        st.markdown(
            "<h2 style='text-align: center; font-size: 22pt; font-weight: 800; color: #003366; font-family: Times New Roman; margin-bottom: 1.15rem;'>(a) Input</h2>",
            unsafe_allow_html=True,
        )
        query_params = st.query_params
        auto_compound = str(query_params.get("compound", "") or "").strip()
        auto_cas = str(query_params.get("cas", "") or "").strip()
        try:
            auto_ph = float(query_params.get("ph", 7.0))
        except Exception:
            auto_ph = 7.0
        auto_run = str(query_params.get("auto_run", "") or "").strip().lower() in {
            "1",
            "true",
            "yes",
        }

        # Oxidant system info box
        st.markdown(
            """
            <div style="background: #f0f4f8;
                        border: 2px solid #003366;
                        border-radius: 6px;
                        padding: 1rem;
                        margin-bottom: 1.15rem;">
                <div style="font-size: 16pt;
                            font-weight: 800;
                            color: #003366;
                            margin-bottom: 0.5rem;
                            font-family: 'Times New Roman', Times, serif;">
                    Oxidant system
                </div>
                <div style="font-size: 15.5pt;
                            font-weight: bold;
                            color: #000;
                            font-family: 'Times New Roman', Times, serif;">
                    Ozone (O₃)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        chemical_name = st.text_input(
            "Chemical name or CAS",
            value=auto_compound,
            placeholder="e.g., acetaminophen",
            label_visibility="visible",
        )
        ph_value = st.number_input(
            "pH",
            min_value=0.0,
            max_value=14.0,
            value=auto_ph,
            step=0.1,
            label_visibility="visible",
        )
        category_choice = st.selectbox(
            "Reaction class override",
            options=["Automatic assignment"] + LOCKED_CATEGORY27_OPTIONS,
            index=0,
        )
        category_input = (
            "" if category_choice == "Automatic assignment" else category_choice
        )

        # Add spacing before button
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        predict_btn = st.button(
            "Run Prediction",
            type="primary",
            disabled=not chemical_name.strip(),
            use_container_width=True,
        ) or (auto_run and bool(chemical_name.strip()))
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        structure_placeholder = st.empty()

    dataset_record = None
    if chemical_name.strip() and os.path.exists(reference_dataset_path):
        try:
            ref_df = load_reference_dataset(reference_dataset_path)
            dataset_record = find_compound_record(chemical_name, ref_df)
        except Exception:
            dataset_record = None

    dataset_source = "Reference dataset" if dataset_record else ""
    preloaded_cas = dataset_record.get("cas") if dataset_record else None
    if not preloaded_cas and auto_cas:
        preloaded_cas = auto_cas
    preloaded_smiles = dataset_record.get("smiles") if dataset_record else None
    preloaded_category27 = (
        dict(dataset_record.get("category27_info"))
        if (dataset_record and dataset_record.get("category27_info"))
        else None
    )
    preview_category27 = preloaded_category27 or normalize_category27_input(
        category_input
    )
    preview_class_label = (
        category27_label_for_display(preview_category27)
        if preview_category27
        else "Automatic assignment"
    )

    with main_cols[1]:
        st.markdown(
            "<h2 style='text-align: center; font-size: 22pt; font-weight: 800; color: #003366; font-family: Times New Roman; margin-bottom: 1.15rem;'>(c) Processing</h2>",
            unsafe_allow_html=True,
        )

        if predict_btn and chemical_name.strip():
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #228b22; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ✓ CAS Retrieval
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #555; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Querying chemical database...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #228b22; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ✓ SMILES Retrieval
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #555; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Fetching molecular structure...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #228b22; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ✓ Fingerprint Generation
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #555; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Computing Morgan fingerprint...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #228b22; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ✓ Classification
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #555; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Assigning reaction class...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #228b22; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ✓ Model Prediction
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #555; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Running Transformer model...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #228b22; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ✓ Prediction Range
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #555; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Checking applicability domain...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            prediction_range_placeholder = st.empty()
        else:
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #999999; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ○ CAS Retrieval
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #b0b0b0; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Pending...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            prediction_range_placeholder = st.empty()
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #999999; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ○ SMILES Retrieval
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #b0b0b0; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Pending...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #999999; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ○ Fingerprint Generation
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #b0b0b0; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Pending...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #999999; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ○ Classification
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #b0b0b0; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Pending...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #999999; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ○ Model Prediction
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #b0b0b0; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Pending...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #999999; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ○ Prediction Range
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #b0b0b0; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Pending...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with main_cols[2]:
        st.markdown(
            "<h2 style='text-align: center; font-size: 22pt; font-weight: 800; color: #003366; font-family: Times New Roman; margin-bottom: 1.15rem;'>(d) Results</h2>",
            unsafe_allow_html=True,
        )

    if not predict_btn:
        # Show placeholder structure in Input column
        with structure_placeholder:
            st.markdown(
                """
                <div style="background: #f8f9fa;
                            border: 2px dashed #ccc;
                            border-radius: 6px;
                            padding: 1.5rem;
                            text-align: center;
                            margin-bottom: 1.15rem;">
                    <div style="font-size: 22pt;
                                font-weight: 800;
                                color: #003366;
                                line-height: 1.15;
                                font-family: 'Times New Roman', Times, serif;">
                        (b) Structure
                    </div>
                    <div style="font-size: 12pt;
                                color: #bbb;
                                font-family: 'Times New Roman', Times, serif;
                                margin-top: 0.5rem;">
                        Run prediction to confirm the resolved structure
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with main_cols[2]:
            st.markdown(
                """
                <div style="background: #f8f9fa;
                            border: 2px dashed #ccc;
                            border-radius: 8px;
                            padding: 2rem;
                            text-align: center;
                            margin-top: 2rem;">
                    <div style="font-size: 18pt;
                                color: #999;
                                font-family: 'Times New Roman', Times, serif;
                                margin-bottom: 0.5rem;">
                        Awaiting prediction...
                    </div>
                    <div style="font-size: 12pt;
                                color: #bbb;
                                font-family: 'Times New Roman', Times, serif;">
                        Enter compound information and click "Run Prediction"
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    step_events = [
        {
            "index": 1,
            "title": "Compound query",
            "state": "Success",
            "tone": "success",
            "message": f"Received compound input '{chemical_name.strip()}' at pH {ph_value:.1f}.",
        }
    ]
    cas = preloaded_cas
    cas_source = dataset_source if cas else ""
    smiles = preloaded_smiles
    smiles_source = dataset_source if smiles else ""
    category27_info = preloaded_category27

    def render_failure_view(message: str):
        render_pipeline_board(step_events)
        result_cols = st.columns([1.05, 0.95], gap="medium")
        with result_cols[0]:
            render_detail_card(
                "Workflow Snapshot",
                [
                    ("Compound", chemical_name.strip()),
                    ("Resolved CAS", cas or "Not found"),
                    ("Resolved SMILES", smiles or "Not found"),
                    (
                        "Reaction class",
                        (
                            category27_label_for_display(category27_info)
                            if category27_info
                            else "Not assigned"
                        ),
                    ),
                ],
            )
        with result_cols[1]:
            render_alert_banner(message, tone="error")

    with st.spinner("Running the prediction workflow..."):
        if cas:
            step_events.append(
                {
                    "index": 2,
                    "title": "CAS retrieval",
                    "state": "Success",
                    "tone": "success",
                    "message": f"Matched CAS from the reference dataset: {cas}.",
                }
            )
        else:
            cas, cas_source = get_cas_number(chemical_name)
            if cas:
                step_events.append(
                    {
                        "index": 2,
                        "title": "CAS retrieval",
                        "state": "Success",
                        "tone": "success",
                        "message": f"Retrieved CAS via {cas_source}: {cas}.",
                    }
                )
            else:
                step_events.append(
                    {
                        "index": 2,
                        "title": "CAS retrieval",
                        "state": "Fallback",
                        "tone": "warn",
                        "message": "No CAS record was found, so the workflow continues with direct structure retrieval.",
                    }
                )

        if smiles:
            step_events.append(
                {
                    "index": 3,
                    "title": "Structure retrieval",
                    "state": "Success",
                    "tone": "success",
                    "message": "Loaded SMILES from the reference dataset.",
                }
            )
        else:
            smiles = get_smiles_via_api(chemical_name, cas_number=cas)
            if smiles:
                smiles_source = "Online lookup"
                step_events.append(
                    {
                        "index": 3,
                        "title": "Structure retrieval",
                        "state": "Success",
                        "tone": "success",
                        "message": "Retrieved a valid SMILES string through online lookup.",
                    }
                )
            else:
                step_events.append(
                    {
                        "index": 3,
                        "title": "Structure retrieval",
                        "state": "Error",
                        "tone": "error",
                        "message": "No reliable SMILES string was found for the current compound.",
                    }
                )
                render_failure_view(
                    "The workflow stopped because no reliable SMILES string could be obtained from the reference dataset or online retrieval."
                )
                return

        # Display molecular structure in Input column
        if smiles:
            rendered_structure = False
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    img = Draw.MolToImage(mol, size=(320, 220))
                    img_buf = io.BytesIO()
                    img.save(img_buf, format="PNG")
                    img_b64 = base64.b64encode(img_buf.getvalue()).decode("ascii")
                    with structure_placeholder:
                        st.markdown(
                            f"""
                            <div style="background: #ffffff;
                                        border: 1px solid rgba(0, 51, 102, 0.18);
                                        border-radius: 6px;
                                        padding: 1rem 0.9rem;
                                        margin-top: 0.5rem;
                                        min-height: 244px;
                                        text-align: center;">
                                <div style="font-size: 22pt;
                                            font-weight: 800;
                                            color: #003366;
                                            font-family: 'Times New Roman', Times, serif;
                                            line-height: 1.15;
                                            margin-bottom: 0.55rem;">
                                    (b) Structure
                                </div>
                                <img src="data:image/png;base64,{img_b64}"
                                     style="display:block; width:100%; max-width:280px; height:auto; margin:0 auto;" />
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    rendered_structure = True
            except Exception:
                rendered_structure = False
            if not rendered_structure:
                with structure_placeholder:
                    render_alert_banner(
                        "Structure rendering failed. Please verify the resolved SMILES in the Results panel.",
                        tone="warn",
                    )

        category27_info = (
            classify_category27_from_smiles(smiles, manual_input=category_input)
            or category27_info
        )
        if not category27_info:
            step_events.append(
                {
                    "index": 4,
                    "title": "Class assignment",
                    "state": "Error",
                    "tone": "error",
                    "message": "Automatic class assignment did not return a valid reaction class.",
                }
            )
            render_failure_view(
                "The workflow stopped because the reaction class could not be assigned from the current structure or the optional class hint."
            )
            return

        step_events.append(
            {
                "index": 4,
                "title": "Class assignment",
                "state": "Success",
                "tone": "success",
                "message": f"Assigned reaction class: {category27_label_for_display(category27_info)}.",
            }
        )

        target_fp_size = FP_SIZE
        try:
            params_preview = json.loads(
                RUNTIME_CONFIG_PATH.read_text(encoding="utf-8")
            )["params"]
            target_fp_size = int(params_preview["fp_bits"])
        except (OSError, KeyError, TypeError, ValueError):
            pass

        fp = dataset_record.get("fingerprint") if dataset_record else None
        if fp is not None and int(np.asarray(fp).reshape(-1).size) == int(
            target_fp_size
        ):
            fp = np.asarray(fp, dtype=np.float32).reshape(-1)
            step_events.append(
                {
                    "index": 5,
                    "title": "Fingerprint encoding",
                    "state": "Success",
                    "tone": "success",
                    "message": f"Reused the matched {len(fp)}-bit fingerprint directly from the dataset.",
                }
            )
        else:
            if fp is not None:
                step_events.append(
                    {
                        "index": 5,
                        "title": "Fingerprint encoding",
                        "state": "Fallback",
                        "tone": "warn",
                        "message": f"The matched fingerprint length did not fit the required {target_fp_size} bits, so a new fingerprint was rebuilt.",
                    }
                )
            fp = smiles_to_fingerprint(smiles, fp_size=target_fp_size)
            if fp is None:
                step_events.append(
                    {
                        "index": 5,
                        "title": "Fingerprint encoding",
                        "state": "Error",
                        "tone": "error",
                        "message": "Fingerprint generation failed for the available structure.",
                    }
                )
                render_failure_view(
                    "The workflow stopped because RDKit could not construct a valid fingerprint from the retrieved SMILES string."
                )
                return
            if not any(
                event.get("index") == 5 and event.get("tone") == "warn"
                for event in step_events
            ):
                step_events.append(
                    {
                        "index": 5,
                        "title": "Fingerprint encoding",
                        "state": "Success",
                        "tone": "success",
                        "message": f"Generated a {target_fp_size}-bit Morgan fingerprint from the resolved structure.",
                    }
                )

        bundle, t_params, load_err = load_predictor()
        if bundle is None:
            step_events.append(
                {
                    "index": 6,
                    "title": "logk prediction",
                    "state": "Error",
                    "tone": "error",
                    "message": f"Prediction bundle loading failed: {load_err}",
                }
            )
            render_failure_view(
                f"The workflow stopped because the prediction bundle could not be loaded: {load_err}"
            )
            return

        fp_bits_required = int(bundle["fp_bits"])
        if int(np.asarray(fp).reshape(-1).size) != fp_bits_required:
            fp = smiles_to_fingerprint(smiles, fp_size=fp_bits_required)
            if fp is None:
                step_events.append(
                    {
                        "index": 6,
                        "title": "logk prediction",
                        "state": "Error",
                        "tone": "error",
                        "message": "The final model-compatible fingerprint could not be reconstructed.",
                    }
                )
                render_failure_view(
                    "The workflow stopped because the model-compatible fingerprint could not be reconstructed from the retrieved structure."
                )
                return

        logk_pred, detail = predict(
            bundle=bundle,
            fp_dense=fp,
            ph_value=ph_value,
            category27_info=category27_info,
            canonical_smiles=canonicalize_smiles(smiles)[0],
        )
        step_events.append(
            {
                "index": 6,
                "title": "logk prediction",
                "state": "Success",
                "tone": "success",
                "message": "The prediction model completed inference and returned the final logk estimate.",
            }
        )
        range_source = str(detail.get("AD_source", "") or "")
        step_events.append(
            {
                "index": 7,
                "title": "prediction range",
                "state": "Online assessment",
                "tone": (
                    "success"
                    if str(detail.get("DECAT_AD_label", "")) == "In AD"
                    else (
                        "warn"
                        if "Borderline" in str(detail.get("DECAT_AD_label", ""))
                        else "error"
                    )
                ),
                "message": range_source
                or "Applicability was assessed against the training fingerprint space.",
            }
        )
        with prediction_range_placeholder.container():
            render_prediction_range_card(detail)

    class_line = category27_label_for_display(category27_info)
    assignment_basis = category27_reason_for_display(category27_info)

    # Display results in the third column
    with main_cols[2]:
        # Chemical name
        st.markdown(
            f"""
            <div style="background: #f8f9fa;
                        border-left: 4px solid #003366;
                        padding: 0.9rem 1rem;
                        margin-bottom: 0.8rem;
                        min-height: 106px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;">
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #003366;
                            line-height: 1.2;
                            margin-bottom: 0.35rem;
                            font-family: 'Times New Roman', Times, serif;">
                    Chemical
                </div>
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #000;
                            line-height: 1.25;
                            font-family: 'Times New Roman', Times, serif;">
                    {escape(chemical_name.strip())}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # CAS
        st.markdown(
            f"""
            <div style="background: #f8f9fa;
                        border-left: 4px solid #003366;
                        padding: 0.9rem 1rem;
                        margin-bottom: 0.8rem;
                        min-height: 106px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;">
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #003366;
                            line-height: 1.2;
                            margin-bottom: 0.35rem;
                            font-family: 'Times New Roman', Times, serif;">
                    CAS
                </div>
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #000;
                            line-height: 1.25;
                            font-family: 'Times New Roman', Times, serif;">
                    {escape(cas if cas else 'Not found')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # k and Logk (prominent display with red color)
        k_value = 10**logk_pred if logk_pred is not None else None
        st.markdown(
            f"""
            <div style="background: #fff5f5;
                        border-left: 4px solid #dc3545;
                        padding: 0.9rem 1rem;
                        margin-bottom: 0.8rem;
                        min-height: 142px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;">
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #003366;
                            line-height: 1.2;
                            margin-bottom: 0.5rem;
                            font-family: 'Times New Roman', Times, serif;">
                    Rate Constant
                </div>
                <div style="font-size: 23pt;
                            font-weight: 800;
                            color: #dc3545;
                            font-family: 'Times New Roman', Times, serif;
                            line-height: 1.22;
                            margin-bottom: 0.3rem;">
                    k = {escape(f"{k_value:.4e}" if k_value is not None else "N/A")} M⁻¹s⁻¹
                </div>
                <div style="font-size: 23pt;
                            font-weight: 800;
                            color: #dc3545;
                            line-height: 1.22;
                            font-family: 'Times New Roman', Times, serif;">
                    logk = {escape(f"{logk_pred:.4f}" if logk_pred is not None else "N/A")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Class
        st.markdown(
            f"""
            <div style="background: #f8f9fa;
                        border-left: 4px solid #003366;
                        padding: 0.9rem 1rem;
                        margin-bottom: 0.8rem;
                        min-height: 106px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;">
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #003366;
                            line-height: 1.2;
                            margin-bottom: 0.35rem;
                            font-family: 'Times New Roman', Times, serif;">
                    Class
                </div>
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #000;
                            line-height: 1.25;
                            font-family: 'Times New Roman', Times, serif;">
                    {escape(class_line)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # pH
        st.markdown(
            f"""
            <div style="background: #f8f9fa;
                        border-left: 4px solid #003366;
                        padding: 0.9rem 1rem;
                        margin-bottom: 0.8rem;
                        min-height: 106px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;">
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #003366;
                            line-height: 1.2;
                            margin-bottom: 0.35rem;
                            font-family: 'Times New Roman', Times, serif;">
                    pH
                </div>
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #000;
                            line-height: 1.25;
                            font-family: 'Times New Roman', Times, serif;">
                    {escape(f"{ph_value:.1f}")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # SMILES
        st.markdown(
            f"""
            <div style="background: #f8f9fa;
                        border-left: 4px solid #003366;
                        padding: 0.9rem 1rem;
                        margin-bottom: 0.8rem;
                        min-height: 106px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;">
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #003366;
                            line-height: 1.2;
                            margin-bottom: 0.35rem;
                            font-family: 'Times New Roman', Times, serif;">
                    SMILES
                </div>
                <div style="font-size: 17pt;
                            font-weight: 800;
                            color: #000;
                            font-family: 'Times New Roman', Times, serif;
                            line-height: 1.25;
                            word-break: break-all;">
                    {escape(smiles or 'Not found')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _running_in_streamlit():
    """Detect if stream is launched via `streamlit run`.
    Uses runtime context to avoid AttributeError on older Streamlit versions.
    """
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if not _running_in_streamlit():
        script_path = os.path.abspath(__file__)
        print("Starting the prediction app with streamlit run...")
        try:
            subprocess.run(
                [sys.executable, "-m", "streamlit", "run", script_path], check=False
            )
        except FileNotFoundError:
            print(
                "Streamlit is unavailable. Install the deployment requirements first."
            )
        raise SystemExit(0)
    main()
