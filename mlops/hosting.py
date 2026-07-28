"""
mlops/hosting.py
----------------
STAGE 4 of the MLOps pipeline: promote the registered model to the live Space.

WHY THIS IS A DELIBERATE, SEPARATE STEP
Training a better model and SHIPPING that model are different decisions. This
stage is what turns a registered artefact into the one real users hit, so it
performs three safety checks before doing anything irreversible:

  1. The registered model actually loads.
  2. It predicts sensibly - a risky order must score above a safe one.
  3. It is not materially worse than the model already deployed.

Only then does it copy the model and its metadata into the Space repository,
which triggers a rebuild.

The third check is the important one. An automated pipeline that blindly
promotes whatever it just trained will eventually promote a regression. Here a
drop of more than 0.02 ROC-AUC stops the promotion and leaves the current model
serving traffic.

Run it with:
    venv\\Scripts\\python.exe mlops\\hosting.py
"""

import json
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# How much worse the new model may be before promotion is refused.
MAX_ROC_AUC_REGRESSION = 0.02

# Two contrasting orders used to prove the model still behaves sensibly.
SAFE_ORDER = {
    "estimated_delivery_days": 40.0, "distance_km": 15.0,
    "customer_lat": -23.55, "customer_lng": -46.63,
    "seller_lat": -23.56, "seller_lng": -46.64,
    "same_state": 1, "is_sao_paulo_seller": 1,
    "total_price": 60.0, "total_freight": 13.0, "freight_ratio": 0.216,
    "price_per_item": 60.0, "freight_per_kg": 32.5,
    "total_payment_value": 73.0, "max_installments": 1, "n_payment_methods": 1,
    "n_items": 1, "n_distinct_products": 1, "n_sellers": 1, "multi_seller_order": 0,
    "total_weight_g": 400.0, "max_volume_cm3": 2000.0, "weight_per_item_g": 400.0,
    "avg_photos_qty": 3.0, "avg_description_len": 800.0,
    "purchase_month": 6, "purchase_day_of_week": 4, "purchase_hour": 15,
    "purchase_week_of_year": 24, "purchase_is_weekend": 0,
    "customer_state": "SP", "seller_state": "SP",
    "primary_category": "stationery", "payment_type": "credit_card",
}
RISKY_ORDER = {
    **SAFE_ORDER,
    "estimated_delivery_days": 7.0, "distance_km": 2692.7,
    "customer_lat": -3.10, "customer_lng": -60.02,
    "same_state": 0, "customer_state": "AM",
    "purchase_month": 11, "purchase_week_of_year": 47,
    "primary_category": "furniture_decor",
}


def main() -> int:
    config.banner("STAGE 4  |  PROMOTE THE REGISTERED MODEL TO THE LIVE SPACE")

    api = HfApi(token=config.token())
    order = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES

    # ---- fetch the candidate ----------------------------------------------
    print(f"  Downloading the registered model from {config.MODEL_REPO}")
    model_path = hf_hub_download(
        repo_id=config.MODEL_REPO, filename=config.MODEL_FILE,
        repo_type="model", token=config.token())
    metadata_path = hf_hub_download(
        repo_id=config.MODEL_REPO, filename=config.METADATA_FILE,
        repo_type="model", token=config.token())
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    candidate_auc = metadata["test_metrics_shipped"]["ROC-AUC"]
    print(f"  Candidate: {metadata['model_name']}, ROC-AUC {candidate_auc}, "
          f"trained {metadata['trained_at']}")

    # ---- check 1: it loads -------------------------------------------------
    config.banner("Pre-flight checks")
    model = joblib.load(model_path)
    print("  [PASS] model file loads")

    # ---- check 2: it predicts sensibly ------------------------------------
    safe = float(model.predict_proba(pd.DataFrame([SAFE_ORDER])[order])[0, 1])
    risky = float(model.predict_proba(pd.DataFrame([RISKY_ORDER])[order])[0, 1])
    print(f"  [{'PASS' if risky > safe else 'FAIL'}] ranks sensibly: "
          f"risky {risky * 100:.1f}% vs safe {safe * 100:.1f}%")
    if risky <= safe:
        print("\n  Refusing to promote: the model rates a safe order as riskier.")
        return 1

    # ---- check 3: no material regression ----------------------------------
    try:
        live_metadata_path = hf_hub_download(
            repo_id=config.SPACE_REPO, filename=config.METADATA_FILE,
            repo_type="space", token=config.token())
        live_auc = json.loads(
            Path(live_metadata_path).read_text(encoding="utf-8")
        )["test_metrics_shipped"]["ROC-AUC"]
        delta = candidate_auc - live_auc
        print(f"  Currently deployed ROC-AUC : {live_auc}")
        print(f"  Candidate ROC-AUC          : {candidate_auc}  ({delta:+.4f})")
        if delta < -MAX_ROC_AUC_REGRESSION:
            print(f"\n  [FAIL] Regression of {abs(delta):.4f} exceeds the "
                  f"{MAX_ROC_AUC_REGRESSION} tolerance.")
            print("  Refusing to promote. The live Space keeps its current model.")
            return 1
        print(f"  [PASS] within the {MAX_ROC_AUC_REGRESSION} regression tolerance")
    except Exception:
        # No model deployed yet, so there is nothing to regress against.
        print("  [SKIP] no model currently deployed - nothing to compare against")

    # ---- promote -----------------------------------------------------------
    config.banner("Promoting to the Space")
    message = (f"Promote {metadata['model_name']} "
               f"(ROC-AUC {candidate_auc}, threshold {metadata['decision_threshold']})")

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        # Copy through a staging folder so both files land in a single commit,
        # which prevents the Space rebuilding against a half-updated pair.
        (staged / config.MODEL_FILE).write_bytes(Path(model_path).read_bytes())
        (staged / config.METADATA_FILE).write_bytes(Path(metadata_path).read_bytes())

        api.upload_folder(
            folder_path=str(staged),
            repo_id=config.SPACE_REPO,
            repo_type="space",
            commit_message=message,
        )

    print(f"  Uploaded {config.MODEL_FILE} and {config.METADATA_FILE}")
    print(f"  The Space is now rebuilding its Docker image.")

    config.banner("STAGE 4 COMPLETE")
    print(f"  Promoted : {metadata['model_name']} @ ROC-AUC {candidate_auc}")
    print(f"  Live at  : https://huggingface.co/spaces/{config.SPACE_REPO}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(config.run_stage(main, config.SPACE_REPO, "space"))
