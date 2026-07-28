"""
app.py
------
Olist Delivery Risk Radar - a Streamlit web application.

WHAT STREAMLIT IS
-----------------
Streamlit turns a plain Python script into a web page. There is no HTML, no
JavaScript and no web server to configure. You write `st.button("Click me")`
and a real button appears in the browser. Every time the user touches
anything, Streamlit simply re-runs this file from top to bottom and redraws
the page with the new values.

THE FOUR TABS
-------------
  Dashboard      - interactive analytics over all 96,470 historical orders
  Predict        - score a single new order at checkout
  Model Insights - how the model was built and how well it performs
  About          - project information and links

Run it locally with:
    venv\\Scripts\\streamlit.exe run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# PAGE SETUP  -  must be the first Streamlit command in the file
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Olist Delivery Risk Radar",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Paths are built relative to this file so the app works identically on a
# laptop and inside a Hugging Face Space, where the folder layout differs.
APP_DIR = Path(__file__).resolve().parent

COLOURS = {
    "ontime": "#2E86AB",
    "late": "#D64545",
    "accent": "#F4A259",
    "grey": "#6C757D",
    "green": "#2E9E5B",
}
RISK_SEQUENCE = ["#2E86AB", "#7FB3D5", "#F4A259", "#E07A5F", "#D64545"]
EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# LOADING  -  cached so the files are read from disk only once
# ---------------------------------------------------------------------------
# @st.cache_resource is for things you load once and reuse, like a model.
# @st.cache_data is for data tables. Without these, Streamlit would reload
# everything on every single click and the app would crawl.
@st.cache_resource(show_spinner="Loading the model...")
def load_model():
    model = joblib.load(APP_DIR / "model.joblib")
    metadata = json.loads((APP_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    return model, metadata


@st.cache_data(show_spinner="Loading historical orders...")
def load_dashboard_data() -> pd.DataFrame:
    return pd.read_parquet(APP_DIR / "dashboard_data.parquet")


@st.cache_data(show_spinner=False)
def load_config() -> dict:
    return json.loads((APP_DIR / "app_config.json").read_text(encoding="utf-8"))


MODEL, META = load_model()
CONFIG = load_config()
THRESHOLD = META["decision_threshold"]
BASE_RATE = META["positive_class_rate"]
REFERENCE = CONFIG["reference"]


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    """Great-circle distance in km - the same formula used during training."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


# ===========================================================================
# HEADER
# ===========================================================================
st.title("📦 Olist Delivery Risk Radar")
st.markdown(
    f"**Predicting late deliveries on Brazil's largest e-commerce marketplace** "
    f"&nbsp;·&nbsp; {REFERENCE['total_orders']:,} historical orders "
    f"({REFERENCE['date_range'][0]} to {REFERENCE['date_range'][1]}) "
    f"&nbsp;·&nbsp; model: {META['model_name']}, ROC-AUC "
    f"{META['test_metrics_shipped']['ROC-AUC']:.3f}"
)

tab_dashboard, tab_predict, tab_model, tab_about = st.tabs(
    ["📊  Dashboard", "🔮  Predict an Order", "📈  Model Insights", "ℹ️  About"]
)


# ===========================================================================
# TAB 1  |  DASHBOARD
# ===========================================================================
with tab_dashboard:
    df = load_dashboard_data()

    st.subheader("Filter the data")
    filter_row = st.columns([1.1, 1.1, 1.4, 1.4])

    with filter_row[0]:
        months = sorted(df["purchase_month"].unique())
        month_range = st.select_slider(
            "Purchase period",
            options=months,
            value=(months[0], months[-1]),
            format_func=lambda d: pd.Timestamp(d).strftime("%b %Y"),
        )
    with filter_row[1]:
        state_options = sorted(df["customer_state"].dropna().unique().tolist())
        picked_states = st.multiselect("Destination state", state_options, default=[])
    with filter_row[2]:
        category_options = sorted(df["primary_category"].dropna().unique().tolist())
        picked_categories = st.multiselect("Product category", category_options, default=[])
    with filter_row[3]:
        payment_options = sorted(df["payment_type"].dropna().unique().tolist())
        picked_payments = st.multiselect("Payment method", payment_options, default=[])

    # Apply every filter the user has set.
    view = df[(df["purchase_month"] >= month_range[0]) & (df["purchase_month"] <= month_range[1])]
    if picked_states:
        view = view[view["customer_state"].isin(picked_states)]
    if picked_categories:
        view = view[view["primary_category"].isin(picked_categories)]
    if picked_payments:
        view = view[view["payment_type"].isin(picked_payments)]

    # If the filters exclude everything we fall back to the full dataset rather
    # than calling st.stop(). st.stop() halts the WHOLE script, which would
    # leave the other three tabs completely blank - a confusing dead end.
    if view.empty:
        st.warning(
            "No orders match that combination of filters, so the charts below "
            "show **all** orders instead. Try widening your selection."
        )
        view = df

    # ---- KPI tiles --------------------------------------------------------
    st.subheader("Key numbers")
    late_rate = view["is_late"].mean()
    revenue = view["total_price"].sum()
    revenue_at_risk = view.loc[view["is_late"] == 1, "total_price"].sum()

    kpi = st.columns(5)
    kpi[0].metric("Orders", f"{len(view):,}",
                  delta=f"{len(view) / len(df) * 100:.0f}% of all orders",
                  delta_color="off")
    kpi[1].metric("Late rate", f"{late_rate * 100:.2f}%",
                  delta=f"{(late_rate - BASE_RATE) * 100:+.2f} pp vs overall",
                  delta_color="inverse")
    kpi[2].metric("Revenue", f"R$ {revenue / 1_000_000:.2f}M")
    kpi[3].metric("Revenue delivered late", f"R$ {revenue_at_risk / 1000:,.0f}K",
                  delta=f"{revenue_at_risk / revenue * 100:.1f}% of revenue",
                  delta_color="off")
    kpi[4].metric("Median journey", f"{view['distance_km'].median():,.0f} km")

    st.divider()

    # ---- Row 1: trend over time + map ------------------------------------
    left, right = st.columns([1.25, 1])

    with left:
        st.markdown("##### Order volume and late rate over time")
        monthly = (
            view.groupby("purchase_month")
            .agg(orders=("is_late", "size"), late_rate=("is_late", "mean"))
            .reset_index()
        )
        monthly = monthly[monthly["orders"] >= 30]
        monthly["late_rate"] *= 100

        fig = go.Figure()
        fig.add_bar(x=monthly["purchase_month"], y=monthly["orders"],
                    name="Orders", marker_color=COLOURS["ontime"], opacity=0.65)
        fig.add_scatter(x=monthly["purchase_month"], y=monthly["late_rate"],
                        name="Late rate (%)", yaxis="y2", mode="lines+markers",
                        line={"color": COLOURS["late"], "width": 3})
        fig.update_layout(
            yaxis={"title": "Orders"},
            yaxis2={"title": "Late rate (%)", "overlaying": "y", "side": "right",
                    "showgrid": False},
            legend={"orientation": "h", "y": 1.12, "x": 0},
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            height=380, hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("##### Where deliveries fail")
        by_state = (
            view.groupby("customer_state", observed=True)
            .agg(orders=("is_late", "size"), late_rate=("is_late", "mean"))
            .reset_index()
        )
        by_state = by_state[by_state["orders"] >= 20]
        by_state["late_rate"] *= 100
        coords = CONFIG["state_coordinates"]
        by_state["lat"] = by_state["customer_state"].map(
            lambda s: coords.get(s, {}).get("lat"))
        by_state["lng"] = by_state["customer_state"].map(
            lambda s: coords.get(s, {}).get("lng"))
        by_state = by_state.dropna(subset=["lat", "lng"])

        fig = px.scatter_geo(
            by_state, lat="lat", lon="lng",
            size="orders", color="late_rate",
            hover_name="customer_state",
            hover_data={"orders": ":,", "late_rate": ":.2f", "lat": False, "lng": False},
            color_continuous_scale=RISK_SEQUENCE,
            size_max=38, scope="south america",
            labels={"late_rate": "Late %"},
        )
        fig.update_geos(fitbounds="locations", showcountries=True,
                        countrycolor="white", showland=True, landcolor="#EDF2F4")
        fig.update_layout(margin={"l": 0, "r": 0, "t": 10, "b": 0}, height=380)
        st.plotly_chart(fig, width="stretch")
        st.caption("Bubble size = order volume · colour = late-delivery rate")

    st.divider()

    # ---- Row 2: three drivers --------------------------------------------
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("##### Distance drives risk")
        banded = view.dropna(subset=["distance_km"]).copy()
        banded["band"] = pd.cut(
            banded["distance_km"], [0, 50, 150, 400, 800, 1500, 5000],
            labels=["0-50", "50-150", "150-400", "400-800", "800-1500", "1500+"],
        )
        grouped = (banded.groupby("band", observed=True)["is_late"].mean() * 100).reset_index()
        fig = px.bar(grouped, x="band", y="is_late", color="is_late",
                     color_continuous_scale=RISK_SEQUENCE,
                     labels={"band": "Distance (km)", "is_late": "Late rate (%)"})
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=330,
                          margin={"l": 10, "r": 10, "t": 20, "b": 10})
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.markdown("##### A tight promise is a broken promise")
        promised = view[view["estimated_delivery_days"].between(0, 60)].copy()
        promised["band"] = pd.cut(
            promised["estimated_delivery_days"], [0, 7, 14, 21, 30, 60],
            labels=["0-7", "8-14", "15-21", "22-30", "31-60"],
        )
        grouped = (promised.groupby("band", observed=True)["is_late"].mean() * 100).reset_index()
        fig = px.bar(grouped, x="band", y="is_late", color="is_late",
                     color_continuous_scale=RISK_SEQUENCE,
                     labels={"band": "Days promised at checkout", "is_late": "Late rate (%)"})
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=330,
                          margin={"l": 10, "r": 10, "t": 20, "b": 10})
        st.plotly_chart(fig, width="stretch")

    with c3:
        st.markdown("##### Riskiest product categories")
        by_category = (
            view.groupby("primary_category", observed=True)
            .agg(orders=("is_late", "size"), late_rate=("is_late", "mean"))
            .reset_index()
        )
        by_category = by_category[by_category["orders"] >= 50].nlargest(10, "late_rate")
        by_category["late_rate"] *= 100
        fig = px.bar(by_category.sort_values("late_rate"), x="late_rate",
                     y="primary_category", orientation="h", color="late_rate",
                     color_continuous_scale=RISK_SEQUENCE,
                     labels={"late_rate": "Late rate (%)", "primary_category": ""})
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=330,
                          margin={"l": 10, "r": 10, "t": 20, "b": 10})
        st.plotly_chart(fig, width="stretch")

    st.divider()

    # ---- Row 3: the commercial impact ------------------------------------
    st.markdown("##### 💰 Why this matters: late delivery destroys customer reviews")
    impact_left, impact_right = st.columns([1.4, 1])

    scored = view.dropna(subset=["review_score"])
    with impact_left:
        if scored.empty:
            st.info("No review data in the current selection.")
        else:
            share = (
                pd.crosstab(scored["is_late"], scored["review_score"], normalize="index")
                .mul(100).stack().rename("pct").reset_index()
            )
            share["Outcome"] = share["is_late"].map({0: "On time", 1: "Late"})
            fig = px.bar(share, x="review_score", y="pct", color="Outcome",
                         barmode="group",
                         color_discrete_map={"On time": COLOURS["ontime"],
                                             "Late": COLOURS["late"]},
                         labels={"review_score": "Customer review score",
                                 "pct": "Share of orders (%)"})
            fig.update_layout(height=330, margin={"l": 10, "r": 10, "t": 20, "b": 10},
                              legend={"orientation": "h", "y": 1.15, "x": 0})
            st.plotly_chart(fig, width="stretch")

    with impact_right:
        if not scored.empty:
            avg = scored.groupby("is_late")["review_score"].mean()
            bad = scored.assign(bad=scored["review_score"] <= 2).groupby("is_late")["bad"].mean()
            st.metric("Average stars - delivered on time", f"{avg.get(0, float('nan')):.2f} ⭐")
            st.metric("Average stars - delivered late", f"{avg.get(1, float('nan')):.2f} ⭐",
                      delta=f"{avg.get(1, 0) - avg.get(0, 0):.2f} stars", delta_color="inverse")
            if bad.get(0, 0) > 0:
                st.metric("1-2 star reviews when late", f"{bad.get(1, 0) * 100:.1f}%",
                          delta=f"{bad.get(1, 0) / bad.get(0, 1):.1f}x the on-time rate",
                          delta_color="inverse")
            st.caption(
                "Every late delivery we can predict in advance is a customer "
                "relationship we still have time to protect."
            )


# ===========================================================================
# TAB 2  |  PREDICT
# ===========================================================================
with tab_predict:
    st.subheader("Score a single order at checkout")
    st.markdown(
        "Fill in what the shop knows **the moment the customer clicks Buy**, then press "
        "**Assess delivery risk**. The model returns a calibrated probability - if it says "
        "20%, then roughly 1 in 5 orders like this really do arrive late."
    )

    # --- quick presets ----------------------------------------------------
    st.markdown("**Quick examples**")
    preset_cols = st.columns(4)
    # Four presets chosen to land in each of the four risk bands, so the tier
    # system can be demonstrated end to end in a few clicks.
    PRESETS = {
        "typical": {"label": "🟢 Typical order",
                    "help": "Sao Paulo to Sao Paulo, 20-day promise, quiet month"},
        "moderate": {"label": "🟡 Long-haul, fair promise",
                     "help": "Sao Paulo to Pernambuco with a reasonable 12-day window"},
        "aggressive": {"label": "🟠 Aggressive promise",
                       "help": "Local delivery, but promised in only 6 days"},
        "blackfriday": {"label": "🔴 Black Friday long-haul",
                        "help": "Sao Paulo to Amazonas in the busiest week of the year"},
    }
    for column, (key, info) in zip(preset_cols, PRESETS.items()):
        if column.button(info["label"], help=info["help"], width="stretch"):
            st.session_state["preset"] = key

    preset = st.session_state.get("preset", "typical")
    defaults = {
        "typical": dict(seller="SP", customer="SP", promise=20, price=90.0, freight=17.0,
                        items=1, sellers=1, weight=750.0, month=6, day=15,
                        category="housewares", payment="credit_card", installments=2),
        "moderate": dict(seller="SP", customer="PE", promise=12, price=250.0, freight=28.0,
                         items=1, sellers=1, weight=1200.0, month=8, day=20,
                         category="health_beauty", payment="credit_card", installments=3),
        "aggressive": dict(seller="SP", customer="SP", promise=6, price=250.0, freight=28.0,
                           items=1, sellers=1, weight=1200.0, month=8, day=20,
                           category="health_beauty", payment="credit_card", installments=3),
        "blackfriday": dict(seller="SP", customer="AM", promise=12, price=480.0, freight=42.0,
                            items=1, sellers=1, weight=2000.0, month=11, day=24,
                            category="furniture_decor", payment="credit_card", installments=8),
    }[preset]

    states = sorted(CONFIG["state_coordinates"].keys())
    categories = CONFIG["product_categories"]
    payments = CONFIG["payment_types"]

    with st.form("prediction_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**🗺️ Route**")
            seller_state = st.selectbox("Seller is in", states,
                                        index=states.index(defaults["seller"]))
            customer_state = st.selectbox("Delivering to", states,
                                          index=states.index(defaults["customer"]))
            promised_days = st.slider("Days promised at checkout", 1, 60,
                                      defaults["promise"],
                                      help="The delivery date shown to the customer")

        with col_b:
            st.markdown("**📦 The order**")
            category = st.selectbox("Product category", categories,
                                    index=categories.index(defaults["category"])
                                    if defaults["category"] in categories else 0)
            order_value = st.number_input("Order value (R$)", 1.0, 15000.0,
                                          defaults["price"], step=10.0)
            freight_value = st.number_input("Freight charged (R$)", 0.0, 500.0,
                                            defaults["freight"], step=1.0)
            weight_g = st.number_input("Total weight (grams)", 1.0, 40000.0,
                                       defaults["weight"], step=100.0)

        with col_c:
            st.markdown("**🛒 Basket and payment**")
            n_items = st.number_input("Number of items", 1, 20, defaults["items"])
            n_sellers = st.number_input("Number of sellers", 1, 5, defaults["sellers"])
            payment_type = st.selectbox("Payment method", payments,
                                        index=payments.index(defaults["payment"])
                                        if defaults["payment"] in payments else 0)
            installments = st.number_input("Instalments", 1, 24, defaults["installments"])
            purchase_date = st.date_input(
                "Purchase date",
                value=pd.Timestamp(2018, defaults["month"], defaults["day"]).date(),
                help="Only the month and week are used - the year is deliberately ignored",
            )

        submitted = st.form_submit_button("🔍  Assess delivery risk",
                                          width="stretch", type="primary")

    if submitted:
        # ---- turn the form answers into the 34 columns the model expects ----
        seller_coords = CONFIG["state_coordinates"][seller_state]
        customer_coords = CONFIG["state_coordinates"][customer_state]

        # How far will this parcel really travel?
        #
        # We do NOT simply measure between the two state centres. That would say
        # a Sao Paulo -> Sao Paulo delivery travels 0 km, when in reality the
        # state is 250 km across. Instead we look up the median distance
        # actually observed on this exact route in the historical data, and fall
        # back to the straight-line calculation only for routes never seen.
        route_key = f"{seller_state}|{customer_state}"
        distance = CONFIG["route_distances"].get(route_key)
        if distance is None:
            distance = haversine_km(seller_coords["lat"], seller_coords["lng"],
                                    customer_coords["lat"], customer_coords["lng"])

        stamp = pd.Timestamp(purchase_date)
        silent = CONFIG["silent_defaults"]

        order = {
            "estimated_delivery_days": float(promised_days),
            "distance_km": distance,
            "customer_lat": customer_coords["lat"], "customer_lng": customer_coords["lng"],
            "seller_lat": seller_coords["lat"], "seller_lng": seller_coords["lng"],
            "same_state": int(seller_state == customer_state),
            "is_sao_paulo_seller": int(seller_state == "SP"),
            "total_price": float(order_value),
            "total_freight": float(freight_value),
            "freight_ratio": float(freight_value) / max(float(order_value), 0.01),
            "price_per_item": float(order_value) / n_items,
            "freight_per_kg": float(freight_value) / max(float(weight_g), 1.0) * 1000,
            "total_payment_value": float(order_value) + float(freight_value),
            "max_installments": int(installments),
            "n_payment_methods": silent["n_payment_methods"],
            "n_items": int(n_items),
            "n_distinct_products": int(n_items),
            "n_sellers": int(n_sellers),
            "multi_seller_order": int(n_sellers > 1),
            "total_weight_g": float(weight_g),
            "max_volume_cm3": silent["max_volume_cm3"],
            "weight_per_item_g": float(weight_g) / n_items,
            "avg_photos_qty": silent["avg_photos_qty"],
            "avg_description_len": silent["avg_description_len"],
            "purchase_month": stamp.month,
            "purchase_day_of_week": stamp.dayofweek,
            "purchase_hour": 15,
            "purchase_week_of_year": int(stamp.isocalendar().week),
            "purchase_is_weekend": int(stamp.dayofweek >= 5),
            "customer_state": customer_state,
            "seller_state": seller_state,
            "primary_category": category,
            "payment_type": payment_type,
        }

        columns = META["numeric_features"] + META["categorical_features"]
        probability = float(MODEL.predict_proba(pd.DataFrame([order])[columns])[0, 1])
        lift = probability / BASE_RATE

        # ---- risk banding ------------------------------------------------
        if probability >= 0.30:
            tier, colour, icon = "VERY HIGH RISK", COLOURS["late"], "🔴"
            action = ("Do not promise this date. Extend the estimate, upgrade to express "
                      "shipping, or contact the seller before confirming the order.")
        elif probability >= THRESHOLD:
            tier, colour, icon = "HIGH RISK", COLOURS["accent"], "🟠"
            action = ("Flag for the proactive-care queue. Send a realistic expectation "
                      "email and monitor dispatch closely.")
        elif probability >= BASE_RATE:
            tier, colour, icon = "MODERATE RISK", "#C9A227", "🟡"
            action = "Slightly above average. Standard monitoring is sufficient."
        else:
            tier, colour, icon = "LOW RISK", COLOURS["green"], "🟢"
            action = "Safe to confirm. No intervention needed."

        st.divider()
        result_left, result_right = st.columns([1, 1.25])

        with result_left:
            st.markdown(
                f"<div style='background:{colour};color:white;padding:22px;"
                f"border-radius:12px;text-align:center'>"
                f"<div style='font-size:15px;opacity:.9'>{icon} PREDICTION</div>"
                f"<div style='font-size:52px;font-weight:800;line-height:1.1'>"
                f"{probability * 100:.1f}%</div>"
                f"<div style='font-size:15px;opacity:.9'>chance of arriving late</div>"
                f"<div style='font-size:20px;font-weight:700;margin-top:10px'>{tier}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.metric("Compared with a typical order",
                      f"{lift:.1f}x", delta=f"average is {BASE_RATE * 100:.1f}%",
                      delta_color="off")
            st.info(f"**Recommended action** — {action}")

        with result_right:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%", "font": {"size": 42}},
                title={"text": "Late-delivery probability", "font": {"size": 15}},
                gauge={
                    "axis": {"range": [0, 60]},
                    "bar": {"color": colour, "thickness": 0.75},
                    "steps": [
                        {"range": [0, BASE_RATE * 100], "color": "#E8F4EA"},
                        {"range": [BASE_RATE * 100, THRESHOLD * 100], "color": "#FDF3D8"},
                        {"range": [THRESHOLD * 100, 30], "color": "#FBE3D6"},
                        {"range": [30, 60], "color": "#F7D0D0"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3},
                                  "thickness": 0.85, "value": THRESHOLD * 100},
                },
            ))
            gauge.update_layout(height=300, margin={"l": 20, "r": 20, "t": 50, "b": 10})
            st.plotly_chart(gauge, width="stretch")
            st.caption(
                f"The black line marks the action threshold of {THRESHOLD * 100:.1f}%, "
                f"chosen to minimise expected business cost."
            )

        # ---- explain the drivers -----------------------------------------
        st.markdown("##### What is driving this score?")
        drivers = []
        if promised_days <= 7:
            drivers.append(("🔴", f"The {promised_days}-day promise is very tight. "
                                  f"Historically 31% of such orders arrive late."))
        elif promised_days <= 14:
            drivers.append(("🟠", f"A {promised_days}-day promise leaves limited slack."))
        else:
            drivers.append(("🟢", f"A {promised_days}-day promise is generous "
                                  f"(median is {REFERENCE['median_estimated_days']:.0f} days)."))

        if distance > 1500:
            drivers.append(("🔴", f"{distance:,.0f} km is a long haul. Journeys over "
                                  f"1,500 km are late 13.1% of the time versus 6.5% under 50 km."))
        elif distance > 400:
            drivers.append(("🟡", f"{distance:,.0f} km is above the median journey of "
                                  f"{REFERENCE['median_distance_km']:,.0f} km."))
        else:
            drivers.append(("🟢", f"{distance:,.0f} km is a short journey."))

        if seller_state == customer_state:
            drivers.append(("🟢", f"Seller and customer are both in {seller_state}, "
                                  f"so the parcel stays within one state "
                                  f"(typical journey {distance:,.0f} km)."))
        else:
            drivers.append(("🟡", f"The parcel crosses from {seller_state} to "
                                  f"{customer_state}, a typical journey of "
                                  f"{distance:,.0f} km."))

        if stamp.month == 11:
            drivers.append(("🔴", "November carries the Black Friday surge - the late rate "
                                  "hit 14.3% in November 2017."))
        elif stamp.month in (2, 3):
            drivers.append(("🟠", "February and March 2018 saw a sustained network "
                                  "disruption, peaking at 21.4%."))

        for icon_mark, text in drivers:
            st.markdown(f"{icon_mark} &nbsp; {text}")

        with st.expander("See the exact values sent to the model"):
            # The order dictionary mixes numbers and text (e.g. 2693.0 and "AM").
            # A table column has to hold one single type, so we format every
            # value into a readable string before displaying it.
            rows = []
            for name in columns:
                value = order[name]
                pretty = f"{value:,.2f}" if isinstance(value, float) else str(value)
                rows.append({"feature": name, "value": pretty})
            st.dataframe(pd.DataFrame(rows), width="stretch", height=420,
                         hide_index=True)


# ===========================================================================
# TAB 3  |  MODEL INSIGHTS
# ===========================================================================
with tab_model:
    st.subheader("How the model was built")

    info = st.columns(4)
    shipped = META["test_metrics_shipped"]
    info[0].metric("Winning model", META["model_name"])
    info[1].metric("ROC-AUC", f"{shipped['ROC-AUC']:.3f}")
    info[2].metric("Recall (late orders caught)", f"{shipped['Recall'] * 100:.1f}%")
    info[3].metric("Action threshold", f"{THRESHOLD * 100:.1f}%")

    st.divider()

    st.markdown("##### Model comparison")
    st.markdown(
        "Three models from three different families were trained and compared on the same "
        "held-out 20% of orders that none of them ever saw during training."
    )
    comparison = pd.DataFrame(META["all_models"]).T
    comparison = comparison[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]]
    st.dataframe(
        comparison.style.format("{:.3f}").background_gradient(subset=["ROC-AUC"], cmap="Blues"),
        width="stretch",
    )
    st.caption(
        "A model that always guessed 'on time' would score 91.9% accuracy and 0.500 ROC-AUC "
        "while catching zero late deliveries. That is why accuracy is not our headline metric."
    )

    st.divider()

    left, right = st.columns(2)
    with left:
        st.markdown("##### What the model relies on")
        importance = pd.DataFrame(META["top_features"])
        fig = px.bar(importance.sort_values("importance"), x="importance", y="feature",
                     orientation="h", color="importance",
                     color_continuous_scale=RISK_SEQUENCE,
                     labels={"importance": "Drop in ROC-AUC when shuffled", "feature": ""})
        fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False,
                          margin={"l": 10, "r": 10, "t": 20, "b": 10})
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("##### Preventing data leakage")
        st.markdown(
            """
**Data leakage** means letting the model see facts that would not exist yet at the
moment we need the prediction. It is like letting a student see the exam answers:
the score looks perfect and means nothing.

Our rule: **the model may only use what the shop knows at checkout.**

These columns were deliberately removed:

| Column | Why it was banned |
|---|---|
| `order_delivered_customer_date` | The target is built from it |
| `order_delivered_carrier_date` | Recorded days after checkout |
| `order_approved_at` | Recorded after payment clears |
| `order_status` | Only known once the order finishes |
| `review_score` | Written after delivery |
| `purchase_year` | Would not generalise to future orders |

Every preprocessing step — filling gaps, scaling, encoding — is fitted on the
**training half only** and lives inside the saved pipeline, so no information
from the test set can influence the model.
            """
        )

    st.divider()
    st.markdown("##### Honest limitations")
    st.warning(
        f"**Precision is {shipped['Precision'] * 100:.0f}%.** Roughly {(1 - shipped['Precision']) * 10:.0f} "
        f"in 10 flagged orders would actually have arrived on time. This is an acceptable "
        f"trade-off only because the intervention is cheap — a courtesy email costs far less "
        f"than the 1-star review that 54% of late deliveries attract. The threshold can be "
        f"raised if intervention costs rise."
    )
    st.caption(
        f"Trained {META['trained_at']} · {META['n_training_rows']:,} training orders · "
        f"{META['n_test_rows']:,} test orders · calibration: {META['calibration']} · "
        f"scikit-learn {META['library_versions']['scikit-learn']}"
    )


# ===========================================================================
# TAB 4  |  ABOUT
# ===========================================================================
with tab_about:
    st.subheader("About this project")

    about_left, about_right = st.columns([1.5, 1])

    with about_left:
        st.markdown(
            f"""
#### The problem

Olist is a Brazilian marketplace connecting thousands of small sellers to customers
across all 27 states. When a seller delivers late, **the customer blames Olist** —
and our analysis of {REFERENCE['total_orders']:,} orders shows exactly how expensive
that is:

- Orders delivered **on time** average **4.29 stars**
- Orders delivered **late** average **2.57 stars**
- The share of 1–2 star reviews jumps from **9.2% to 54.1%** — a **5.9× increase**

#### The solution

This app scores every order **at the moment it is placed**, using only information
available at checkout. Orders flagged as high-risk can be given a realistic delivery
estimate, upgraded shipping, or a proactive courtesy contact — *before* the customer
is disappointed.

#### How it works

1. **Nine raw Kaggle tables** were joined into one modelling table of
   {REFERENCE['total_orders']:,} delivered orders
2. **Distance in kilometres** between seller and buyer was engineered from
   1,000,163 GPS readings using the haversine formula — this column does not
   exist in the raw data
3. **Three models** were trained and compared; {META['model_name']} won
4. **Isotonic calibration** was applied so the percentages shown are honest
5. The **entire pipeline** — imputation, scaling, encoding, model, calibration —
   is saved in a single `model.joblib` file
            """
        )

    with about_right:
        st.markdown(
            """
#### Dataset

**Brazilian E-Commerce Public Dataset by Olist**
Real, anonymised commercial data.
Licence: CC BY-NC-SA 4.0

[View on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

#### Team

- Kartik Joshi
- Prem Kukreja
- Samuel Alex
- Gagandeep Singh
- Aditya Chitale
- Karan Baid

#### Course

**Software Development for AI Models**
Instructor: Prof. JP Agarwal
Masters in AI with Business

#### Links

[GitHub repository](https://github.com/moikukreja/olist-delivery-risk)

[Hugging Face Space](https://huggingface.co/spaces/samuelalex37/olist-delivery-risk)
            """
        )

    st.divider()
    st.caption(
        "Built with Streamlit and scikit-learn · deployed automatically to Hugging Face "
        "Spaces via GitHub Actions on every push to main."
    )
