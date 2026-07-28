"""
mlops/config.py
---------------
Settings shared by every stage of the MLOps pipeline.

WHY A SEPARATE FILE
Each pipeline stage runs as its own GitHub Actions job on its own fresh
machine. They never share memory, only files on the Hugging Face Hub. Keeping
the repository names and column lists in one place means a stage can never
disagree with the stage before it about where the data lives or what it
contains.
"""

import os

# ---------------------------------------------------------------------------
# HUGGING FACE REPOSITORIES
# ---------------------------------------------------------------------------
# The Hub acts as the pipeline's storage layer, exactly as in the course
# material. Three repositories, each with a distinct job:
#
#   DATASET  the versioned data the pipeline trains on
#   MODEL    the versioned trained artefact
#   SPACE    the running application
#
# Splitting them means the model can be retrained and re-registered without
# touching the app, and the app can be redeployed without retraining.
HF_USER = os.getenv("HF_USER", "samuelalex37")

DATASET_REPO = f"{HF_USER}/olist-delivery-risk-data"
MODEL_REPO = f"{HF_USER}/olist-delivery-risk-model"
SPACE_REPO = f"{HF_USER}/olist-delivery-risk"

# ---------------------------------------------------------------------------
# FILE NAMES INSIDE THE DATASET REPOSITORY
# ---------------------------------------------------------------------------
ANALYTICAL_FILE = "olist_analytical.parquet"
XTRAIN_FILE = "Xtrain.parquet"
XTEST_FILE = "Xtest.parquet"
YTRAIN_FILE = "ytrain.csv"
YTEST_FILE = "ytest.csv"

MODEL_FILE = "model.joblib"
METADATA_FILE = "model_metadata.json"

# ---------------------------------------------------------------------------
# MODELLING CONTRACT
# ---------------------------------------------------------------------------
# These lists are the single source of truth for what the model may see. They
# are duplicated nowhere else in the pipeline.
TARGET = "is_late"

NUMERIC_FEATURES = [
    "estimated_delivery_days",
    "distance_km", "customer_lat", "customer_lng", "seller_lat", "seller_lng",
    "same_state", "is_sao_paulo_seller",
    "total_price", "total_freight", "freight_ratio", "price_per_item",
    "freight_per_kg", "total_payment_value", "max_installments", "n_payment_methods",
    "n_items", "n_distinct_products", "n_sellers", "multi_seller_order",
    "total_weight_g", "max_volume_cm3", "weight_per_item_g",
    "avg_photos_qty", "avg_description_len",
    "purchase_month", "purchase_day_of_week", "purchase_hour",
    "purchase_week_of_year", "purchase_is_weekend",
]

CATEGORICAL_FEATURES = [
    "customer_state", "seller_state", "primary_category", "payment_type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that would leak the answer. Listed explicitly so the training stage
# can assert none of them ever reaches the model.
BANNED_COLUMNS = [
    "order_delivered_customer_date", "order_delivered_carrier_date",
    "order_approved_at", "order_estimated_delivery_date", "order_status",
    "actual_delivery_days", "days_late", "review_score", "purchase_year",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20

# The cost of one missed late delivery expressed in false alarms. Drives the
# decision threshold; see the report, Section 5.5.
COST_RATIO = 5


def token() -> str:
    """Read the Hugging Face token from the environment, or fail loudly."""
    value = os.getenv("HF_TOKEN")
    if not value:
        raise SystemExit(
            "HF_TOKEN is not set.\n"
            "  In GitHub Actions it comes from the encrypted repository secret.\n"
            "  Locally, set it for one command:  $env:HF_TOKEN='hf_...'"
        )
    return value


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def explain_permission_error(error: Exception, repo_id: str, repo_type: str) -> int:
    """
    Turn a Hub 403 into instructions instead of a stack trace.

    A fine-grained token scoped to a single Space cannot create or write other
    repositories. That is correct and deliberate - but the raw error is sixty
    lines of traceback that buries the one sentence that matters, so we catch it
    and say plainly what needs to change.
    """
    text = str(error)
    if "403" not in text and "Forbidden" not in text:
        return -1  # not a permissions problem; let the caller re-raise

    print()
    print("=" * 78)
    print("  PERMISSION DENIED")
    print("=" * 78)
    print(f"  The token authenticated, but it is not allowed to write to")
    print(f"  {repo_type} '{repo_id}'.")
    print()
    print("  This is expected if you are still using the Space-scoped token from")
    print("  the deployment workflow. The pipeline needs access to three")
    print("  repositories, not one.")
    print()
    print("  FIX (recommended - keeps least privilege):")
    print(f"    1. Create the repositories on huggingface.co/new-dataset and")
    print(f"       huggingface.co/new  (model):")
    print(f"         - {DATASET_REPO}   (dataset)")
    print(f"         - {MODEL_REPO}  (model)")
    print("    2. Edit the token at huggingface.co/settings/tokens")
    print("    3. Add BOTH new repositories to its selected-repositories list,")
    print("       with read and write ticked, alongside the existing Space.")
    print()
    print("  QUICKER ALTERNATIVE (broader access):")
    print("    Tick 'Write access to contents/settings of your repos' under")
    print("    User permissions. This lets the token create repositories on")
    print("    demand, but grants it authority over the whole account.")
    print("=" * 78)
    return 1


def run_stage(main_fn, repo_id: str, repo_type: str) -> int:
    """
    Run a pipeline stage, converting a Hub permission failure into guidance.

    Wrapping the whole stage rather than each individual call means a 403 is
    reported the same helpful way no matter which Hub operation triggered it.
    """
    try:
        return main_fn()
    except Exception as exc:  # noqa: BLE001 - deliberately broad, then re-raised
        code = explain_permission_error(exc, repo_id, repo_type)
        if code >= 0:
            return code
        raise
