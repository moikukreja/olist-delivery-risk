"""
02_build_dataset.py
-------------------
STEP 2: turn nine separate tables into ONE clean table we can model on.

Think of the raw data like nine sheets in an Excel workbook. Each sheet holds a
different piece of the story: one has the orders, one has the customers, one has
the products, one has GPS coordinates for every Brazilian postcode.

A machine-learning model cannot read nine sheets. It needs one flat table where
every row is one order and every column is one fact about that order.

This script does that joining, and it also CREATES NEW FACTS that were not in
the raw data at all - most importantly the real physical distance in kilometres
between the seller and the buyer.

Run it with:
    venv\\Scripts\\python.exe src\\02_build_dataset.py
"""

import sys

import numpy as np
import pandas as pd

from utils import (
    PROCESSED_DIR,
    banner,
    configure_pandas,
    load_raw,
    sub_banner,
)

# Brazil's real geographic bounding box. Any GPS point outside this rectangle is
# a data-entry error (a few rows place Brazilian postcodes in Africa or Asia).
BRAZIL_LAT_MIN, BRAZIL_LAT_MAX = -33.75, 5.28
BRAZIL_LNG_MIN, BRAZIL_LNG_MAX = -73.99, -34.79

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lng1, lat2, lng2):
    """
    Distance in kilometres between two points on Earth.

    Why not simple subtraction? Because the Earth is a ball, not a flat map.
    Going "one degree east" is a much shorter distance near the poles than at
    the equator. The haversine formula is the standard way to measure the true
    curved distance along the surface. It is exactly what a courier's route
    planner uses as a first estimate.
    """
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------------------
# STEP 2.1  -  base table: delivered orders only, with the target attached
# ----------------------------------------------------------------------------
def build_base_orders() -> pd.DataFrame:
    banner("STEP 2.1  |  BASE TABLE  -  delivered orders + the target variable")

    orders = load_raw("orders")
    print(f"  Starting rows (all orders)            : {len(orders):,}")

    # We can only learn from orders that actually finished. An order still in
    # transit has no known outcome, and a cancelled order was never delivered.
    delivered = orders[orders["order_status"] == "delivered"].copy()
    print(f"  After keeping only 'delivered'        : {len(delivered):,}")

    delivered = delivered.dropna(
        subset=["order_delivered_customer_date", "order_estimated_delivery_date"]
    )
    print(f"  After dropping rows with missing dates: {len(delivered):,}")

    # THE TARGET: did the parcel arrive after the date we promised?
    delivered["is_late"] = (
        delivered["order_delivered_customer_date"]
        > delivered["order_estimated_delivery_date"]
    ).astype(int)

    # Kept only for the dashboard and for EDA. These are NEVER given to the model.
    delivered["actual_delivery_days"] = (
        delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    delivered["days_late"] = (
        delivered["order_delivered_customer_date"] - delivered["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    print(f"\n  Late orders : {delivered['is_late'].sum():,} "
          f"({delivered['is_late'].mean() * 100:.2f}%)")
    return delivered


# ----------------------------------------------------------------------------
# STEP 2.2  -  what was actually in the box?
# ----------------------------------------------------------------------------
def build_item_features() -> pd.DataFrame:
    banner("STEP 2.2  |  ORDER CONTENTS  -  items, products and categories")

    items = load_raw("order_items")
    products = load_raw("products")
    translation = load_raw("category_translation")

    print(f"  order_items rows : {len(items):,}  (one row per physical item)")
    print(f"  products rows    : {len(products):,}")

    # Translate the 71 Portuguese category names into English so the app and the
    # report are readable by an English-speaking audience.
    products = products.merge(translation, on="product_category_name", how="left")
    products["category"] = (
        products["product_category_name_english"]
        .fillna(products["product_category_name"])
        .fillna("unknown")
    )

    # Physical size of the box, in cubic centimetres.
    products["product_volume_cm3"] = (
        products["product_length_cm"] * products["product_height_cm"] * products["product_width_cm"]
    )

    items = items.merge(
        products[
            [
                "product_id",
                "category",
                "product_weight_g",
                "product_volume_cm3",
                "product_photos_qty",
                "product_description_lenght",
            ]
        ],
        on="product_id",
        how="left",
    )

    # An order can contain several items. The model needs ONE row per order, so
    # we summarise: how many items, how much they cost, how heavy in total.
    sub_banner("2.2a  Collapsing many items into one row per order")
    agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_distinct_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        max_item_price=("price", "max"),
        total_weight_g=("product_weight_g", "sum"),
        max_volume_cm3=("product_volume_cm3", "max"),
        avg_photos_qty=("product_photos_qty", "mean"),
        avg_description_len=("product_description_lenght", "mean"),
    )

    # For category and seller we take the MOST EXPENSIVE item in the order,
    # because that item dominates the packaging, handling and shipping route.
    primary = (
        items.sort_values("price", ascending=False)
        .drop_duplicates(subset="order_id", keep="first")
        .set_index("order_id")[["category", "seller_id"]]
        .rename(columns={"category": "primary_category", "seller_id": "primary_seller_id"})
    )

    result = agg.join(primary).reset_index()
    print(f"  Result: {len(result):,} orders x {result.shape[1]} columns")
    return result


# ----------------------------------------------------------------------------
# STEP 2.3  -  how did the customer pay?
# ----------------------------------------------------------------------------
def build_payment_features() -> pd.DataFrame:
    banner("STEP 2.3  |  PAYMENT BEHAVIOUR")

    payments = load_raw("payments")
    print(f"  payments rows : {len(payments):,}  (an order can be split across cards)")

    agg = payments.groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        n_payment_methods=("payment_sequential", "count"),
    )

    # The payment TYPE we record is the one used for the largest amount.
    main_type = (
        payments.sort_values("payment_value", ascending=False)
        .drop_duplicates(subset="order_id", keep="first")
        .set_index("order_id")[["payment_type"]]
    )

    result = agg.join(main_type).reset_index()
    print(f"  Result: {len(result):,} orders x {result.shape[1]} columns")
    return result


# ----------------------------------------------------------------------------
# STEP 2.4  -  THE STAR FEATURE: how far does the parcel have to travel?
# ----------------------------------------------------------------------------
def build_geolocation_lookup() -> pd.DataFrame:
    banner("STEP 2.4  |  GEOLOCATION  -  building a postcode -> GPS lookup")

    geo = load_raw("geolocation")
    print(f"  Raw geolocation rows : {len(geo):,}")

    # Throw away impossible coordinates before averaging, otherwise one bad row
    # in the middle of the Atlantic drags a whole postcode's average with it.
    inside_brazil = (
        geo["geolocation_lat"].between(BRAZIL_LAT_MIN, BRAZIL_LAT_MAX)
        & geo["geolocation_lng"].between(BRAZIL_LNG_MIN, BRAZIL_LNG_MAX)
    )
    removed = (~inside_brazil).sum()
    geo = geo[inside_brazil]
    print(f"  Removed {removed:,} rows with GPS points outside Brazil's borders")

    # There are ~1 million GPS readings but only ~19,000 postcodes. We collapse
    # them to one representative point each. We use the MEDIAN rather than the
    # average because the median ignores a handful of extreme readings.
    lookup = (
        geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .median()
        .reset_index()
        .rename(
            columns={
                "geolocation_zip_code_prefix": "zip_prefix",
                "geolocation_lat": "lat",
                "geolocation_lng": "lng",
            }
        )
    )
    print(f"  Collapsed to {len(lookup):,} unique postcodes with one GPS point each")
    return lookup


# ----------------------------------------------------------------------------
# STEP 2.5  -  join everything together
# ----------------------------------------------------------------------------
def assemble(base, items_agg, payments_agg, geo_lookup) -> pd.DataFrame:
    banner("STEP 2.5  |  ASSEMBLY  -  joining all the pieces into one table")

    customers = load_raw("customers")
    sellers = load_raw("sellers")

    df = base.merge(customers, on="customer_id", how="left")
    print(f"  + customers  -> {len(df):,} rows")

    df = df.merge(items_agg, on="order_id", how="inner")
    print(f"  + order items-> {len(df):,} rows   (inner join drops orders with no items)")

    df = df.merge(payments_agg, on="order_id", how="left")
    print(f"  + payments   -> {len(df):,} rows")

    df = df.merge(
        sellers.rename(columns={"seller_id": "primary_seller_id"}),
        on="primary_seller_id",
        how="left",
    )
    print(f"  + sellers    -> {len(df):,} rows")

    sub_banner("2.5a  Attaching GPS points and computing distance")
    df = df.merge(
        geo_lookup.rename(columns={"zip_prefix": "customer_zip_code_prefix",
                                   "lat": "customer_lat", "lng": "customer_lng"}),
        on="customer_zip_code_prefix",
        how="left",
    )
    df = df.merge(
        geo_lookup.rename(columns={"zip_prefix": "seller_zip_code_prefix",
                                   "lat": "seller_lat", "lng": "seller_lng"}),
        on="seller_zip_code_prefix",
        how="left",
    )

    df["distance_km"] = haversine_km(
        df["seller_lat"], df["seller_lng"], df["customer_lat"], df["customer_lng"]
    )

    matched = df["distance_km"].notna().mean() * 100
    print(f"  Distance computed for {matched:.2f}% of orders")
    print(f"  Median distance : {df['distance_km'].median():,.0f} km")
    print(f"  Longest journey : {df['distance_km'].max():,.0f} km")
    return df


# ----------------------------------------------------------------------------
# STEP 2.6  -  create the remaining engineered features
# ----------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 2.6  |  FEATURE ENGINEERING  -  creating new facts from old ones")

    purchase = df["order_purchase_timestamp"]

    # --- WHEN was it bought? Seasonality drives courier congestion. ----------
    df["purchase_year"] = purchase.dt.year
    df["purchase_month"] = purchase.dt.month
    df["purchase_day_of_week"] = purchase.dt.dayofweek        # Monday = 0
    df["purchase_hour"] = purchase.dt.hour
    df["purchase_is_weekend"] = (purchase.dt.dayofweek >= 5).astype(int)
    df["purchase_week_of_year"] = purchase.dt.isocalendar().week.astype(int)

    # --- HOW LONG was the promise? ------------------------------------------
    # IMPORTANT AND LEGITIMATE: the promised date is shown to the customer ON
    # the checkout page, so we genuinely know it at prediction time. A generous
    # 30-day promise is easy to keep; a tight 5-day promise is not. This is one
    # of our strongest honest features.
    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] - purchase
    ).dt.total_seconds() / 86400

    # --- MONEY ---------------------------------------------------------------
    df["freight_ratio"] = df["total_freight"] / df["total_price"].replace(0, np.nan)
    df["price_per_item"] = df["total_price"] / df["n_items"]
    df["freight_per_kg"] = df["total_freight"] / df["total_weight_g"].replace(0, np.nan) * 1000

    # --- GEOGRAPHY -----------------------------------------------------------
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    df["is_sao_paulo_seller"] = (df["seller_state"] == "SP").astype(int)

    # --- LOGISTICS COMPLEXITY ------------------------------------------------
    df["multi_seller_order"] = (df["n_sellers"] > 1).astype(int)
    df["weight_per_item_g"] = df["total_weight_g"] / df["n_items"]

    print("  Created the following new columns:")
    created = [
        "purchase_year", "purchase_month", "purchase_day_of_week", "purchase_hour",
        "purchase_is_weekend", "purchase_week_of_year", "estimated_delivery_days",
        "freight_ratio", "price_per_item", "freight_per_kg", "same_state",
        "is_sao_paulo_seller", "multi_seller_order", "weight_per_item_g", "distance_km",
    ]
    for name in created:
        print(f"    +  {name}")
    return df


def main() -> None:
    configure_pandas()

    banner("OLIST DELIVERY-DELAY PREDICTION  |  SCRIPT 02: BUILD THE MODELLING TABLE")

    base = build_base_orders()
    items_agg = build_item_features()
    payments_agg = build_payment_features()
    geo_lookup = build_geolocation_lookup()

    df = assemble(base, items_agg, payments_agg, geo_lookup)
    df = engineer_features(df)

    banner("RESULT  |  THE ANALYTICAL TABLE")
    print(f"  Final shape : {df.shape[0]:,} orders  x  {df.shape[1]} columns")
    print(f"  Late rate   : {df['is_late'].mean() * 100:.2f}%")

    sub_banner("Preview of the key engineered columns")
    preview_cols = [
        "is_late", "estimated_delivery_days", "distance_km", "total_price",
        "total_freight", "n_items", "customer_state", "seller_state", "primary_category",
    ]
    print(df[preview_cols].head(10).to_string(index=False))

    out_path = PROCESSED_DIR / "olist_analytical.parquet"
    df.to_parquet(out_path, index=False)

    banner("SAVED")
    print(f"  {out_path}")
    print(f"  Size on disk: {out_path.stat().st_size / 1024**2:.1f} MB")
    print()


if __name__ == "__main__":
    sys.exit(main())
