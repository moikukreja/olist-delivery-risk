"""
app.py
------
Olist Delivery Risk Radar - FastAPI backend.

WHAT THIS FILE DOES
-------------------
It plays two roles at once:

  1. AN API SERVER. The React front-end asks it questions over HTTP
     ("what are the numbers for Sao Paulo?", "how risky is this order?")
     and it answers with JSON. All the pandas and scikit-learn work happens
     here, on the server, where those libraries actually exist.

  2. A WEB SERVER. Once the React app has been compiled into plain HTML,
     CSS and JavaScript, this file hands those files to the browser too.
     One process, one port - which is exactly what a Hugging Face Docker
     Space expects.

WHY NOT DO THE MATHS IN THE BROWSER?
------------------------------------
The trained model is a scikit-learn pipeline saved as model.joblib. Browsers
cannot run Python. Keeping the model server-side also means the model file is
never shipped to the user, and we can change it without touching the frontend.

Run it locally with:
    venv\\Scripts\\python.exe -m uvicorn app:app --reload --port 7860
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"          # the compiled React app lives here

EARTH_RADIUS_KM = 6371.0

# ---------------------------------------------------------------------------
# LOAD EVERYTHING ONCE, AT STARTUP
# ---------------------------------------------------------------------------
# Reading a 3 MB file on every request would make the app crawl. These are
# module-level, so they are loaded exactly once when the server boots and then
# reused for every visitor.
MODEL = joblib.load(APP_DIR / "model.joblib")
META = json.loads((APP_DIR / "model_metadata.json").read_text(encoding="utf-8"))
CONFIG = json.loads((APP_DIR / "app_config.json").read_text(encoding="utf-8"))
DATA = pd.read_parquet(APP_DIR / "dashboard_data.parquet")

FEATURE_ORDER = META["numeric_features"] + META["categorical_features"]
THRESHOLD = float(META["decision_threshold"])
BASE_RATE = float(META["positive_class_rate"])

STATE_NAMES = {
    "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá", "BA": "Bahia",
    "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás",
    "MA": "Maranhão", "MG": "Minas Gerais", "MS": "Mato Grosso do Sul",
    "MT": "Mato Grosso", "PA": "Pará", "PB": "Paraíba", "PE": "Pernambuco",
    "PI": "Piauí", "PR": "Paraná", "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte", "RO": "Rondônia", "RR": "Roraima",
    "RS": "Rio Grande do Sul", "SC": "Santa Catarina", "SE": "Sergipe",
    "SP": "São Paulo", "TO": "Tocantins",
}

app = FastAPI(
    title="Olist Delivery Risk Radar API",
    description="Predicts whether an e-commerce order will be delivered late.",
    version="1.0.0",
)

# During development the React dev server runs on a different port (5173) from
# this API (7860). Browsers block that by default, so we explicitly allow it.
# In production both are served from the same origin and this never applies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in km - identical to the training-time formula."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


# ===========================================================================
# /api/config  -  everything the front-end needs to draw its forms
# ===========================================================================
@app.get("/api/config")
def get_config() -> dict:
    shipped = META["test_metrics_shipped"]
    return {
        "states": sorted(CONFIG["state_coordinates"].keys()),
        "stateNames": STATE_NAMES,
        "stateCoordinates": CONFIG["state_coordinates"],
        "routeDistances": CONFIG["route_distances"],
        "categories": CONFIG["product_categories"],
        "paymentTypes": CONFIG["payment_types"],
        "reference": CONFIG["reference"],
        "months": sorted(DATA["purchase_month"].dt.strftime("%Y-%m").unique().tolist()),
        "model": {
            "name": META["model_name"],
            "calibration": META["calibration"],
            "trainedAt": META["trained_at"],
            "threshold": THRESHOLD,
            "baseRate": BASE_RATE,
            "trainingRows": META["n_training_rows"],
            "testRows": META["n_test_rows"],
            "sklearnVersion": META["library_versions"]["scikit-learn"],
            "metrics": shipped,
            "comparison": META["all_models"],
            "topFeatures": META["top_features"],
        },
    }


# ===========================================================================
# /api/dashboard  -  aggregate 96,470 orders down to a few hundred numbers
# ===========================================================================
def apply_filters(
    month_from: str | None, month_to: str | None,
    states: list[str] | None, categories: list[str] | None,
    payments: list[str] | None,
) -> pd.DataFrame:
    view = DATA
    if month_from:
        view = view[view["purchase_month"] >= pd.Timestamp(month_from + "-01")]
    if month_to:
        view = view[view["purchase_month"] <= pd.Timestamp(month_to + "-01")]
    if states:
        view = view[view["customer_state"].isin(states)]
    if categories:
        view = view[view["primary_category"].isin(categories)]
    if payments:
        view = view[view["payment_type"].isin(payments)]
    return view


def band_rates(frame: pd.DataFrame, column: str, edges: list, labels: list[str]) -> list[dict]:
    """Group a numeric column into buckets and report the late rate of each."""
    banded = frame.dropna(subset=[column]).copy()
    if banded.empty:
        return []
    banded["band"] = pd.cut(banded[column], bins=edges, labels=labels)
    grouped = banded.groupby("band", observed=True).agg(
        orders=("is_late", "size"), lateRate=("is_late", "mean")
    ).reset_index()
    return [
        {"band": str(row["band"]), "orders": int(row["orders"]),
         "lateRate": round(float(row["lateRate"]) * 100, 2)}
        for _, row in grouped.iterrows()
    ]


@app.get("/api/dashboard")
def get_dashboard(
    monthFrom: str | None = Query(None, description="Earliest month, e.g. 2017-01"),
    monthTo: str | None = Query(None, description="Latest month, e.g. 2018-08"),
    states: list[str] | None = Query(None),
    categories: list[str] | None = Query(None),
    payments: list[str] | None = Query(None),
) -> dict:
    view = apply_filters(monthFrom, monthTo, states, categories, payments)

    # If the filters exclude everything, say so plainly and fall back to the
    # full dataset rather than returning an empty page with no explanation.
    fell_back = False
    if view.empty:
        view, fell_back = DATA, True

    revenue = float(view["total_price"].sum())
    revenue_at_risk = float(view.loc[view["is_late"] == 1, "total_price"].sum())

    # --- headline numbers -------------------------------------------------
    kpis = {
        "orders": int(len(view)),
        "shareOfAllOrders": round(len(view) / len(DATA) * 100, 1),
        "lateRate": round(float(view["is_late"].mean()) * 100, 2),
        "lateOrders": int(view["is_late"].sum()),
        "revenue": round(revenue, 2),
        "revenueAtRisk": round(revenue_at_risk, 2),
        "revenueAtRiskPct": round(revenue_at_risk / revenue * 100, 1) if revenue else 0.0,
        "medianDistance": round(float(view["distance_km"].median()), 0),
        "medianDeliveryDays": round(float(view["actual_delivery_days"].median()), 1),
    }

    # --- month by month ---------------------------------------------------
    monthly = view.groupby("purchase_month").agg(
        orders=("is_late", "size"), lateRate=("is_late", "mean")
    ).reset_index()
    monthly = monthly[monthly["orders"] >= 30]
    monthly_out = [
        {"month": row["purchase_month"].strftime("%Y-%m"),
         "orders": int(row["orders"]),
         "lateRate": round(float(row["lateRate"]) * 100, 2)}
        for _, row in monthly.iterrows()
    ]

    # --- one row per state, for the WebGL map -----------------------------
    by_state = view.groupby("customer_state", observed=True).agg(
        orders=("is_late", "size"), lateRate=("is_late", "mean"),
        revenue=("total_price", "sum"), medianDistance=("distance_km", "median"),
    ).reset_index()
    by_state = by_state[by_state["orders"] >= 10]

    coords = CONFIG["state_coordinates"]
    states_out = []
    for _, row in by_state.iterrows():
        code = str(row["customer_state"])
        position = coords.get(code)
        if not position:
            continue
        states_out.append({
            "code": code,
            "name": STATE_NAMES.get(code, code),
            "lat": position["lat"], "lng": position["lng"],
            "orders": int(row["orders"]),
            "lateRate": round(float(row["lateRate"]) * 100, 2),
            "revenue": round(float(row["revenue"]), 2),
            "medianDistance": round(float(row["medianDistance"]), 0),
        })
    states_out.sort(key=lambda s: s["lateRate"], reverse=True)

    # --- product categories ------------------------------------------------
    by_category = view.groupby("primary_category", observed=True).agg(
        orders=("is_late", "size"), lateRate=("is_late", "mean")
    ).reset_index()
    by_category = by_category[by_category["orders"] >= 40].nlargest(12, "lateRate")
    categories_out = [
        {"name": str(row["primary_category"]), "orders": int(row["orders"]),
         "lateRate": round(float(row["lateRate"]) * 100, 2)}
        for _, row in by_category.iterrows()
    ]

    # --- the commercial impact --------------------------------------------
    scored = view.dropna(subset=["review_score"])
    reviews_out: dict = {"available": False}
    if len(scored) > 50:
        share = pd.crosstab(scored["is_late"], scored["review_score"], normalize="index") * 100
        by_score = []
        for score in sorted(scored["review_score"].unique()):
            by_score.append({
                "score": int(score),
                "onTime": round(float(share.loc[0, score]), 2) if 0 in share.index else 0.0,
                "late": round(float(share.loc[1, score]), 2) if 1 in share.index else 0.0,
            })
        averages = scored.groupby("is_late")["review_score"].mean()
        poor = scored.assign(bad=scored["review_score"] <= 2).groupby("is_late")["bad"].mean()
        reviews_out = {
            "available": True,
            "byScore": by_score,
            "avgOnTime": round(float(averages.get(0, float("nan"))), 2),
            "avgLate": round(float(averages.get(1, float("nan"))), 2),
            "poorOnTime": round(float(poor.get(0, 0)) * 100, 1),
            "poorLate": round(float(poor.get(1, 0)) * 100, 1),
        }

    return {
        "fellBackToAllData": fell_back,
        "kpis": kpis,
        "monthly": monthly_out,
        "states": states_out,
        "categories": categories_out,
        "distanceBands": band_rates(
            view, "distance_km", [0, 50, 150, 400, 800, 1500, 5000],
            ["0-50", "50-150", "150-400", "400-800", "800-1500", "1500+"]),
        "promiseBands": band_rates(
            view, "estimated_delivery_days", [0, 7, 14, 21, 30, 60],
            ["0-7", "8-14", "15-21", "22-30", "31-60"]),
        "reviews": reviews_out,
    }


# ===========================================================================
# /api/predict  -  score one order
# ===========================================================================
class OrderRequest(BaseModel):
    """The 13 things we ask the user for. Everything else is derived."""
    sellerState: str = Field(..., description="Two-letter state code, e.g. SP")
    customerState: str = Field(..., description="Two-letter state code, e.g. AM")
    promisedDays: int = Field(..., ge=1, le=90)
    category: str
    orderValue: float = Field(..., gt=0, le=50000)
    freightValue: float = Field(..., ge=0, le=2000)
    weightGrams: float = Field(..., gt=0, le=100000)
    itemCount: int = Field(1, ge=1, le=50)
    sellerCount: int = Field(1, ge=1, le=10)
    paymentType: str
    installments: int = Field(1, ge=1, le=24)
    purchaseDate: date


def build_driver_list(promised_days: int, distance: float,
                      seller_state: str, customer_state: str, month: int) -> list[dict]:
    """Plain-English reasons behind the score, for the user to sanity-check."""
    drivers: list[dict] = []

    if promised_days <= 7:
        drivers.append({"level": "high", "text":
            f"The {promised_days}-day promise is very tight. Historically 31% of orders "
            f"promised within a week arrive late."})
    elif promised_days <= 14:
        drivers.append({"level": "medium", "text":
            f"A {promised_days}-day promise leaves limited slack."})
    else:
        drivers.append({"level": "low", "text":
            f"A {promised_days}-day promise is generous — the median across all orders "
            f"is {CONFIG['reference']['median_estimated_days']:.0f} days."})

    if distance > 1500:
        drivers.append({"level": "high", "text":
            f"{distance:,.0f} km is a long haul. Journeys beyond 1,500 km are late 13.1% "
            f"of the time, against 6.5% for those under 50 km."})
    elif distance > 400:
        drivers.append({"level": "medium", "text":
            f"{distance:,.0f} km is above the median journey of "
            f"{CONFIG['reference']['median_distance_km']:,.0f} km."})
    else:
        drivers.append({"level": "low", "text": f"{distance:,.0f} km is a short journey."})

    if seller_state == customer_state:
        drivers.append({"level": "low", "text":
            f"Seller and buyer are both in {STATE_NAMES.get(seller_state, seller_state)}, "
            f"so the parcel never leaves the state."})
    else:
        drivers.append({"level": "medium", "text":
            f"The parcel crosses from {STATE_NAMES.get(seller_state, seller_state)} to "
            f"{STATE_NAMES.get(customer_state, customer_state)}."})

    if month == 11:
        drivers.append({"level": "high", "text":
            "November carries the Black Friday surge — the late rate reached 14.3% in "
            "November 2017, roughly triple the preceding months."})
    elif month in (2, 3):
        drivers.append({"level": "medium", "text":
            "February and March 2018 saw a sustained network disruption that peaked at "
            "21.4%, so orders in these months carry extra risk."})

    return drivers


@app.post("/api/predict")
def predict(order: OrderRequest) -> dict:
    coords = CONFIG["state_coordinates"]
    if order.sellerState not in coords or order.customerState not in coords:
        raise HTTPException(status_code=400, detail="Unknown state code.")

    seller = coords[order.sellerState]
    customer = coords[order.customerState]

    # How far will the parcel really travel? We use the median distance actually
    # observed on this exact route, because measuring between two state centres
    # would claim a Sao Paulo -> Sao Paulo delivery travels 0 km. Only if the
    # route was never seen do we fall back to the straight-line calculation.
    route_key = f"{order.sellerState}|{order.customerState}"
    distance = CONFIG["route_distances"].get(route_key)
    if distance is None:
        distance = haversine_km(seller["lat"], seller["lng"],
                                customer["lat"], customer["lng"])

    stamp = pd.Timestamp(order.purchaseDate)
    silent = CONFIG["silent_defaults"]

    features = {
        "estimated_delivery_days": float(order.promisedDays),
        "distance_km": float(distance),
        "customer_lat": customer["lat"], "customer_lng": customer["lng"],
        "seller_lat": seller["lat"], "seller_lng": seller["lng"],
        "same_state": int(order.sellerState == order.customerState),
        "is_sao_paulo_seller": int(order.sellerState == "SP"),
        "total_price": order.orderValue,
        "total_freight": order.freightValue,
        "freight_ratio": order.freightValue / max(order.orderValue, 0.01),
        "price_per_item": order.orderValue / order.itemCount,
        "freight_per_kg": order.freightValue / max(order.weightGrams, 1.0) * 1000,
        "total_payment_value": order.orderValue + order.freightValue,
        "max_installments": order.installments,
        "n_payment_methods": silent["n_payment_methods"],
        "n_items": order.itemCount,
        "n_distinct_products": order.itemCount,
        "n_sellers": order.sellerCount,
        "multi_seller_order": int(order.sellerCount > 1),
        "total_weight_g": order.weightGrams,
        "max_volume_cm3": silent["max_volume_cm3"],
        "weight_per_item_g": order.weightGrams / order.itemCount,
        "avg_photos_qty": silent["avg_photos_qty"],
        "avg_description_len": silent["avg_description_len"],
        "purchase_month": stamp.month,
        "purchase_day_of_week": stamp.dayofweek,
        "purchase_hour": 15,
        "purchase_week_of_year": int(stamp.isocalendar().week),
        "purchase_is_weekend": int(stamp.dayofweek >= 5),
        "customer_state": order.customerState,
        "seller_state": order.sellerState,
        "primary_category": order.category,
        "payment_type": order.paymentType,
    }

    frame = pd.DataFrame([features])[FEATURE_ORDER]
    probability = float(MODEL.predict_proba(frame)[0, 1])

    if probability >= 0.30:
        tier, action = "VERY HIGH", (
            "Do not confirm this delivery date. Extend the estimate, upgrade the "
            "shipping method, or contact the seller before the order is accepted.")
    elif probability >= THRESHOLD:
        tier, action = "HIGH", (
            "Add to the proactive-care queue. Send the customer a realistic "
            "expectation and monitor dispatch closely.")
    elif probability >= BASE_RATE:
        tier, action = "MODERATE", (
            "Slightly above the marketplace average. Standard monitoring is enough.")
    else:
        tier, action = "LOW", "Safe to confirm. No intervention needed."

    return {
        "probability": round(probability, 4),
        "probabilityPct": round(probability * 100, 1),
        "tier": tier,
        "action": action,
        "lift": round(probability / BASE_RATE, 2),
        "threshold": THRESHOLD,
        "baseRate": BASE_RATE,
        "distanceKm": round(float(distance), 1),
        "flagged": probability >= THRESHOLD,
        "drivers": build_driver_list(order.promisedDays, float(distance),
                                     order.sellerState, order.customerState, stamp.month),
        "features": {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in features.items()},
    }


@app.get("/api/health")
def health() -> dict:
    """A simple 'are you alive?' endpoint, used by the Docker health check."""
    return {"status": "ok", "model": META["model_name"], "orders": int(len(DATA))}


# ===========================================================================
# SERVE THE COMPILED REACT APP
# ===========================================================================
# This block must come LAST. It claims every remaining URL, so if it were
# placed above the /api routes it would swallow them.
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_react(full_path: str):
        """
        Hand back the React app for any address that is not an API call.

        React handles its own navigation inside the browser, so the server
        should answer every unknown path with the same index.html and let
        React decide what to show.
        """
        candidate = STATIC_DIR / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(STATIC_DIR / "index.html")
