"""
utils.py
--------
Shared settings and helper functions used by every other script in this project.

Keeping paths in ONE place means that if the project folder ever moves, we only
have to fix it here instead of in five different files.
"""

from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# FOLDER PATHS
# ----------------------------------------------------------------------------
# Path(__file__) = this file.  .parent = the "src" folder.  .parent again = the
# project root folder.  Building paths this way means the project works on any
# computer, no matter where it is saved.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _folder in (PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR):
    _folder.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# THE NINE RAW FILES
# ----------------------------------------------------------------------------
# A dictionary is a lookup table: give it a short nickname, get back a filename.
RAW_FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# Columns that hold dates. Pandas reads everything as text by default, so we
# tell it explicitly which columns to convert into real date/time values.
DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}


def load_raw(name: str) -> pd.DataFrame:
    """Load one raw CSV by its nickname, converting any date columns properly."""
    if name not in RAW_FILES:
        raise KeyError(f"Unknown table '{name}'. Valid names: {list(RAW_FILES)}")

    df = pd.read_csv(RAW_DIR / RAW_FILES[name])

    for column in DATE_COLUMNS.get(name, []):
        df[column] = pd.to_datetime(df[column], errors="coerce")

    return df


def load_all_raw() -> dict[str, pd.DataFrame]:
    """Load all nine raw tables at once and return them in a dictionary."""
    return {name: load_raw(name) for name in RAW_FILES}


# ----------------------------------------------------------------------------
# PRETTY PRINTING HELPERS  (so terminal screenshots look clean in the report)
# ----------------------------------------------------------------------------
LINE_WIDTH = 88


def banner(title: str) -> None:
    """Print a big, obvious section heading."""
    print()
    print("=" * LINE_WIDTH)
    print(f"  {title}")
    print("=" * LINE_WIDTH)


def sub_banner(title: str) -> None:
    """Print a smaller sub-heading."""
    print()
    print(f"--- {title} " + "-" * max(0, LINE_WIDTH - len(title) - 5))


def configure_pandas() -> None:
    """Make pandas print wide tables without cutting columns off."""
    pd.set_option("display.width", LINE_WIDTH * 2)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
