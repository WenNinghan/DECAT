"""Verify the self-contained deployment bundle without training."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


def main() -> int:
    deployment = Path(__file__).resolve().parents[1]
    release = deployment.parent
    app_path = deployment / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("decat_release_streamlit", app_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {app_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    bundle, _, error = module.load_predictor()
    if bundle is None:
        raise RuntimeError(error)
    if len(bundle["models"]) != 3 or len(bundle["residual_models"]) != 1:
        raise RuntimeError(
            "Deployment bundle does not contain Top-3 + one residual-RF model"
        )
    if int(bundle["fp_bits"]) != 3147 or len(bundle["selected_idx"]) != 914:
        raise RuntimeError("Unexpected locked fingerprint dimensions")

    fp = module.smiles_to_fingerprint("CCO", fp_size=int(bundle["fp_bits"]))
    category = module.classify_category27_from_smiles("CCO")
    prediction, detail = module.predict(bundle, fp, 7.0, category, "CCO")
    if not math.isfinite(float(prediction)):
        raise RuntimeError("Prediction is not finite")

    print("Deployment verification PASSED")
    print(f"release={release}")
    print(f"prediction_CCO={float(prediction):.8f}")
    print(f"ad_label_CCO={detail.get('DECAT_AD_label')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
