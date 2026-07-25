"""
01_data_audit.py
----------------
STEP 1 of the machine-learning pipeline: understand the raw data before touching it.

This script answers Task 1.1 of the project brief:
  - What tables do we have and how big are they?
  - What does each column contain?
  - Where are the missing values?
  - Are there duplicates?
  - Do the tables actually join together correctly?
  - What does our target variable (late delivery) look like?

It CHANGES NOTHING. It only looks and reports. That is deliberate: you should
always understand data before you start cleaning it.

Run it with:
    venv\\Scripts\\python.exe src\\01_data_audit.py
"""

import sys

import pandas as pd

from utils import (
    DATE_COLUMNS,
    RAW_FILES,
    REPORTS_DIR,
    banner,
    configure_pandas,
    load_all_raw,
    sub_banner,
)


def audit_table_inventory(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Section A: how many rows and columns does each of the nine tables have?"""
    banner("SECTION A  |  TABLE INVENTORY  -  the nine raw Olist files")

    rows = []
    for name, df in tables.items():
        rows.append(
            {
                "table": name,
                "file": RAW_FILES[name],
                "rows": len(df),
                "columns": df.shape[1],
                "memory_MB": round(df.memory_usage(deep=True).sum() / 1024**2, 1),
            }
        )

    inventory = pd.DataFrame(rows).sort_values("rows", ascending=False)
    print(inventory.to_string(index=False))
    print(f"\n  TOTAL ROWS ACROSS ALL TABLES : {inventory['rows'].sum():,}")
    return inventory


def audit_orders_structure(orders: pd.DataFrame) -> None:
    """Section B: a close look at the orders table - the heart of the dataset."""
    banner("SECTION B  |  THE 'orders' TABLE  -  our central table")

    sub_banner("B1. First 5 rows")
    print(orders.head().to_string())

    sub_banner("B2. Column types and non-null counts")
    orders.info()

    sub_banner("B3. Order status breakdown")
    status = orders["order_status"].value_counts()
    status_pct = (status / len(orders) * 100).round(2)
    print(pd.DataFrame({"count": status, "percent": status_pct}).to_string())

    sub_banner("B4. Time period covered")
    print(f"  Earliest purchase : {orders['order_purchase_timestamp'].min()}")
    print(f"  Latest   purchase : {orders['order_purchase_timestamp'].max()}")


def audit_missing_values(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Section C: find every missing value in every table."""
    banner("SECTION C  |  MISSING VALUES  -  Task 1.1 data-quality check")

    records = []
    for name, df in tables.items():
        missing = df.isna().sum()
        for column, n_missing in missing.items():
            if n_missing > 0:
                records.append(
                    {
                        "table": name,
                        "column": column,
                        "missing": int(n_missing),
                        "pct_missing": round(n_missing / len(df) * 100, 2),
                    }
                )

    if not records:
        print("  No missing values found anywhere.")
        return pd.DataFrame()

    report = pd.DataFrame(records).sort_values("pct_missing", ascending=False)
    print(report.to_string(index=False))

    print(
        "\n  READ THIS AS: a high 'pct_missing' is not automatically bad. "
        "\n  For order dates, a missing value usually means the event never happened"
        "\n  (e.g. the order was cancelled, so it was never delivered)."
    )
    return report


def audit_duplicates(tables: dict[str, pd.DataFrame]) -> None:
    """Section D: are there repeated rows?"""
    banner("SECTION D  |  DUPLICATE ROWS")

    rows = []
    for name, df in tables.items():
        n_dupes = int(df.duplicated().sum())
        rows.append(
            {
                "table": name,
                "duplicate_rows": n_dupes,
                "pct": round(n_dupes / len(df) * 100, 2),
            }
        )

    print(pd.DataFrame(rows).to_string(index=False))
    print(
        "\n  NOTE: the geolocation table is expected to have many duplicates."
        "\n  It stores many GPS readings per postcode, so we will average them later."
    )


def audit_key_integrity(tables: dict[str, pd.DataFrame]) -> None:
    """Section E: check that the tables genuinely link to each other."""
    banner("SECTION E  |  KEY INTEGRITY  -  do the nine tables actually join?")

    orders = tables["orders"]
    items = tables["order_items"]
    customers = tables["customers"]
    products = tables["products"]
    sellers = tables["sellers"]

    checks = [
        (
            "orders.customer_id  ->  customers.customer_id",
            orders["customer_id"].isin(customers["customer_id"]).mean(),
        ),
        (
            "order_items.order_id  ->  orders.order_id",
            items["order_id"].isin(orders["order_id"]).mean(),
        ),
        (
            "order_items.product_id  ->  products.product_id",
            items["product_id"].isin(products["product_id"]).mean(),
        ),
        (
            "order_items.seller_id  ->  sellers.seller_id",
            items["seller_id"].isin(sellers["seller_id"]).mean(),
        ),
    ]

    for label, match_rate in checks:
        verdict = "OK" if match_rate > 0.999 else "PROBLEM"
        print(f"  [{verdict:>7}]  {label:<48}  {match_rate * 100:6.2f}% matched")

    sub_banner("E2. Uniqueness of primary keys")
    print(f"  orders.order_id unique      : {orders['order_id'].is_unique}")
    print(f"  customers.customer_id unique: {customers['customer_id'].is_unique}")
    print(f"  products.product_id unique  : {products['product_id'].is_unique}")
    print(f"  sellers.seller_id unique    : {sellers['seller_id'].is_unique}")

    sub_banner("E3. Orders that have NO items attached")
    orphans = (~orders["order_id"].isin(items["order_id"])).sum()
    print(f"  {orphans:,} orders ({orphans / len(orders) * 100:.2f}%) have no line items.")


def audit_target_preview(orders: pd.DataFrame) -> None:
    """Section F: define and preview the thing we are trying to predict."""
    banner("SECTION F  |  TARGET VARIABLE PREVIEW  -  'was the order late?'")

    print(
        "  DEFINITION"
        "\n  ----------"
        "\n  An order is LATE (target = 1) when the date the customer actually"
        "\n  received it is AFTER the delivery date they were promised at checkout:"
        "\n"
        "\n      is_late = order_delivered_customer_date > order_estimated_delivery_date"
        "\n"
    )

    delivered = orders[orders["order_status"] == "delivered"].copy()
    print(f"  Total orders in dataset            : {len(orders):,}")
    print(f"  Orders with status 'delivered'     : {len(delivered):,}")

    usable = delivered.dropna(
        subset=["order_delivered_customer_date", "order_estimated_delivery_date"]
    ).copy()
    print(f"  ...of which have both dates present: {len(usable):,}")

    usable["is_late"] = (
        usable["order_delivered_customer_date"] > usable["order_estimated_delivery_date"]
    ).astype(int)

    sub_banner("F1. Class balance")
    counts = usable["is_late"].value_counts().sort_index()
    for label, count in counts.items():
        word = "ON TIME" if label == 0 else "LATE   "
        print(f"  {label} = {word} : {count:>7,}  ({count / len(usable) * 100:5.2f}%)")

    late_rate = usable["is_late"].mean()
    print(
        f"\n  => Roughly 1 in {1 / late_rate:.0f} orders arrives late."
        f"\n     This is an IMBALANCED problem, so plain accuracy will be misleading."
        f"\n     A model that always guesses 'on time' would already score "
        f"{(1 - late_rate) * 100:.1f}% accuracy"
        f"\n     while catching ZERO late deliveries - which is useless to the business."
    )

    sub_banner("F2. How late are the late ones?")
    usable["days_late"] = (
        usable["order_delivered_customer_date"] - usable["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400
    late_only = usable.loc[usable["is_late"] == 1, "days_late"]
    print(late_only.describe().to_string())


def audit_leakage_warning() -> None:
    """Section G: name the columns that would cheat, before we ever model."""
    banner("SECTION G  |  DATA-LEAKAGE WARNING  -  Task 1.2 requirement")

    print(
        "  DATA LEAKAGE means letting the model see information that would NOT be"
        "\n  available at the moment we need the prediction. It is like letting a"
        "\n  student see the exam answers: the score looks perfect and means nothing."
        "\n"
        "\n  We need the prediction AT CHECKOUT. So any column recorded AFTER checkout"
        "\n  is banned. The following columns MUST be dropped before modelling:"
        "\n"
    )

    banned = [
        ("order_delivered_customer_date", "This IS the answer - the target is built from it."),
        ("order_estimated_delivery_date", "Used to build the target; keep only as a derived gap."),
        ("order_delivered_carrier_date", "Recorded days after checkout, when the courier collects."),
        ("order_approved_at", "Recorded after payment clears, not known at checkout."),
        ("order_status", "'delivered' / 'cancelled' is only known once the order finishes."),
        ("review_score", "The customer writes the review AFTER delivery."),
        ("shipping_limit_date", "Set by internal logistics after the order is accepted."),
    ]
    for column, reason in banned:
        print(f"    x  {column:<32} {reason}")

    print(
        "\n  WHAT WE ARE ALLOWED TO USE: purchase timestamp, price, freight paid,"
        "\n  product size and weight, product category, payment method, how many items,"
        "\n  and where the buyer and seller are located."
    )


def main() -> None:
    configure_pandas()

    banner("OLIST DELIVERY-DELAY PREDICTION  |  SCRIPT 01: RAW DATA AUDIT")
    print("  Software Development for AI Models  -  Final Group Project")
    print("  Dataset: Brazilian E-Commerce Public Dataset by Olist (Kaggle)")

    print("\n  Loading nine raw CSV files ...")
    tables = load_all_raw()
    print("  Done.")

    inventory = audit_table_inventory(tables)
    audit_orders_structure(tables["orders"])
    missing_report = audit_missing_values(tables)
    audit_duplicates(tables)
    audit_key_integrity(tables)
    audit_target_preview(tables["orders"])
    audit_leakage_warning()

    # Save the two tables we will quote in the report.
    inventory.to_csv(REPORTS_DIR / "table_inventory.csv", index=False)
    missing_report.to_csv(REPORTS_DIR / "missing_values_report.csv", index=False)

    banner("AUDIT COMPLETE")
    print(f"  Saved: {REPORTS_DIR / 'table_inventory.csv'}")
    print(f"  Saved: {REPORTS_DIR / 'missing_values_report.csv'}")
    print()


if __name__ == "__main__":
    sys.exit(main())
