"""
06_build_app_assets.py
----------------------
STEP 6: prepare the two small files the deployed app needs.

WHY THIS SCRIPT EXISTS
----------------------
The full analytical table is 20.6 MB and the raw Kaggle data is 120 MB. We are
not going to upload any of that to Hugging Face - it would make the app slow to
start and the repository huge.

Instead we produce two lean files:

  dashboard_data.parquet  - only the columns the dashboard actually charts,
                            stored in Parquet (a compressed column format that
                            loads far faster than CSV)

  app_config.json         - small lookup tables the prediction form needs:
                            where each Brazilian state sits on the map, the list
                            of product categories, and sensible default values
                            for the few inputs we do not ask the user to type

Run it with:
    venv\\Scripts\\python.exe src\\06_build_app_assets.py
"""

import json
import sys

import numpy as np
import pandas as pd

from utils import (
    PROCESSED_DIR,
    PROJECT_ROOT,
    banner,
    configure_pandas,
    load_raw,
    sub_banner,
)

# Only these columns travel to the cloud. Everything else stays on the laptop.
DASHBOARD_COLUMNS = [
    "order_purchase_timestamp",
    "is_late",
    "days_late",
    "actual_delivery_days",
    "estimated_delivery_days",
    "customer_state",
    "seller_state",
    "primary_category",
    "payment_type",
    "distance_km",
    "total_price",
    "total_freight",
    "n_items",
    "max_installments",
    "same_state",
]

# Inputs the model needs but that a user should not have to type. We fill these
# with the median of the training data - a neutral, typical value.
SILENT_DEFAULT_COLUMNS = [
    "max_volume_cm3",
    "avg_photos_qty",
    "avg_description_len",
    "n_payment_methods",
]


def main() -> int:
    configure_pandas()
    banner("OLIST DELIVERY-DELAY PREDICTION  |  SCRIPT 06: BUILD APP ASSETS")

    path = PROCESSED_DIR / "olist_analytical.parquet"
    if not path.exists():
        print(f"  ERROR: {path} not found. Run 02_build_dataset.py first.")
        return 1

    df = pd.read_parquet(path)
    print(f"  Loaded analytical table: {df.shape[0]:,} orders x {df.shape[1]} columns "
          f"({path.stat().st_size / 1024**2:.1f} MB)")

    # ------------------------------------------------------------------
    # 1. dashboard_data.parquet
    # ------------------------------------------------------------------
    banner("1  |  BUILDING dashboard_data.parquet")

    reviews = load_raw("reviews")[["order_id", "review_score"]].drop_duplicates("order_id")
    slim = df.merge(reviews, on="order_id", how="left")
    slim = slim[DASHBOARD_COLUMNS + ["review_score"]].copy()
    print(f"  Kept {len(slim.columns)} of {df.shape[1]} columns")

    # Storing repeated text as a "category" lets Parquet keep one copy of each
    # distinct value instead of 96,470 copies of the word "housewares".
    for column in ["customer_state", "seller_state", "primary_category", "payment_type"]:
        slim[column] = slim[column].astype("category")

    # Full timestamps are not needed for charting; the month is enough.
    slim["purchase_month"] = slim["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    slim = slim.drop(columns=["order_purchase_timestamp"])

    for column in ["distance_km", "total_price", "total_freight", "days_late",
                   "actual_delivery_days", "estimated_delivery_days"]:
        slim[column] = slim[column].astype("float32")

    out_parquet = PROJECT_ROOT / "dashboard_data.parquet"
    slim.to_parquet(out_parquet, index=False, compression="snappy")
    size_mb = out_parquet.stat().st_size / 1024**2
    print(f"  Saved : {out_parquet.name}")
    print(f"  Size  : {size_mb:.2f} MB  "
          f"(down from {path.stat().st_size / 1024**2:.1f} MB - "
          f"{(1 - size_mb / (path.stat().st_size / 1024**2)) * 100:.0f}% smaller)")

    # ------------------------------------------------------------------
    # 2. app_config.json
    # ------------------------------------------------------------------
    banner("2  |  BUILDING app_config.json")

    sub_banner("2a. Map position of every Brazilian state")
    # The median GPS point of all customers in a state is a good enough centre
    # for plotting a bubble on the map, and it costs us no external map file.
    centroids = (
        df.groupby("customer_state", observed=True)[["customer_lat", "customer_lng"]]
        .median()
        .round(4)
    )
    seller_centroids = (
        df.groupby("seller_state", observed=True)[["seller_lat", "seller_lng"]]
        .median()
        .round(4)
        .rename(columns={"seller_lat": "customer_lat", "seller_lng": "customer_lng"})
    )
    # Some states only ever appear as a seller location, or only as a customer
    # location. Combining both lists means the dropdowns are never missing one.
    all_states = centroids.combine_first(seller_centroids).dropna()
    state_coords = {
        state: {"lat": float(row["customer_lat"]), "lng": float(row["customer_lng"])}
        for state, row in all_states.iterrows()
    }
    print(f"  Mapped {len(state_coords)} states")

    sub_banner("2a-ii. Typical distance for every seller-state -> customer-state route")
    # Using the straight-line distance between two state centres would say that
    # a Sao Paulo -> Sao Paulo delivery travels 0 km, which is never true. Sao
    # Paulo state is 250 km across. So instead we look up the MEDIAN REAL
    # distance actually observed for each route in the historical data.
    route_distances = (
        df.dropna(subset=["distance_km"])
        .groupby(["seller_state", "customer_state"], observed=True)["distance_km"]
        .agg(["median", "size"])
    )
    route_distances = route_distances[route_distances["size"] >= 5]
    routes = {
        f"{seller}|{customer}": round(float(row["median"]), 1)
        for (seller, customer), row in route_distances.iterrows()
    }
    print(f"  Stored {len(routes):,} routes seen at least 5 times")
    same_state_examples = [(k, v) for k, v in routes.items()
                           if k.split("|")[0] == k.split("|")[1]][:5]
    print("  Sanity check - same-state routes are NOT zero:")
    for key, value in same_state_examples:
        print(f"    {key:<10} median {value:,.0f} km")

    sub_banner("2b. Dropdown option lists")
    categories = sorted(df["primary_category"].dropna().unique().tolist())
    payment_types = sorted(df["payment_type"].dropna().unique().tolist())
    print(f"  {len(categories)} product categories, {len(payment_types)} payment types")

    sub_banner("2c. Default values for inputs we do not ask the user for")
    defaults = {col: float(df[col].median()) for col in SILENT_DEFAULT_COLUMNS}
    for name, value in defaults.items():
        print(f"    {name:<24} = {value:,.1f}")

    sub_banner("2d. Reference statistics shown in the app")
    reference = {
        "overall_late_rate": float(df["is_late"].mean()),
        "total_orders": int(len(df)),
        "median_distance_km": float(df["distance_km"].median()),
        "median_order_value": float(df["total_price"].median()),
        "median_freight": float(df["total_freight"].median()),
        "median_estimated_days": float(df["estimated_delivery_days"].median()),
        "median_weight_g": float(df["total_weight_g"].median()),
        "date_range": [
            str(df["order_purchase_timestamp"].min().date()),
            str(df["order_purchase_timestamp"].max().date()),
        ],
    }
    for name, value in reference.items():
        print(f"    {name:<24} = {value}")

    config = {
        "state_coordinates": state_coords,
        "route_distances": routes,
        "product_categories": categories,
        "payment_types": payment_types,
        "silent_defaults": defaults,
        "reference": reference,
    }

    out_json = PROJECT_ROOT / "app_config.json"
    out_json.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\n  Saved : {out_json.name}  ({out_json.stat().st_size / 1024:.0f} KB)")

    # ------------------------------------------------------------------
    banner("DEPLOYMENT PAYLOAD  |  what actually goes to Hugging Face")
    payload = ["app.py", "model.joblib", "model_metadata.json",
               "dashboard_data.parquet", "app_config.json", "requirements.txt", "README.md"]
    total = 0.0
    for name in payload:
        file_path = PROJECT_ROOT / name
        if file_path.exists():
            mb = file_path.stat().st_size / 1024**2
            total += mb
            print(f"    {name:<26} {mb:>7.2f} MB")
        else:
            print(f"    {name:<26} {'(not yet created)':>18}")
    print(f"    {'-' * 26} {'-' * 10}")
    print(f"    {'TOTAL':<26} {total:>7.2f} MB")
    print("\n  Comfortably under Hugging Face's 10 MB-per-file limit, so no Git LFS needed.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
