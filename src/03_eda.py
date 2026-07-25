"""
03_eda.py
---------
STEP 3: Exploratory Data Analysis (EDA).

"Exploratory Data Analysis" just means: look at the data and find out what is
going on before you build anything. It is the equivalent of a doctor examining a
patient before prescribing treatment.

Every chart this script makes is saved automatically as a PNG file inside
reports/figures/. Those PNG files go straight into the LaTeX report - you do
NOT need to take screenshots of them.

The script also prints a set of KEY OBSERVATIONS to the terminal. That printout
IS worth screenshotting, because it is the evidence for the "key observations"
section the project brief asks for.

Run it with:
    venv\\Scripts\\python.exe src\\03_eda.py
"""

import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import (
    FIGURES_DIR,
    PROCESSED_DIR,
    banner,
    configure_pandas,
    load_raw,
    sub_banner,
)

matplotlib.use("Agg")            # draw to files, never open a window
warnings.filterwarnings("ignore")

# A single consistent look for every chart in the report.
PALETTE = {"ontime": "#2E86AB", "late": "#D64545", "accent": "#F4A259", "grey": "#6C757D"}
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "figure.facecolor": "white",
})

OBSERVATIONS: list[str] = []


def note(text: str) -> None:
    """Record a key observation so we can print them all together at the end."""
    OBSERVATIONS.append(text)


def save(fig, filename: str) -> None:
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    print(f"    saved -> reports/figures/{filename}")


# ----------------------------------------------------------------------------
# CHART 1  -  How common is a late delivery?
# ----------------------------------------------------------------------------
def chart_target_balance(df: pd.DataFrame) -> None:
    sub_banner("Chart 1: target balance")
    counts = df["is_late"].value_counts().sort_index()
    late_pct = df["is_late"].mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    bars = axes[0].bar(
        ["On time", "Late"], counts.values,
        color=[PALETTE["ontime"], PALETTE["late"]], width=0.55,
    )
    axes[0].set_title("Delivery outcome (count of orders)")
    axes[0].set_ylabel("Orders")
    for bar, value in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}",
                     ha="center", va="bottom", fontweight="bold")

    axes[1].pie(
        counts.values, labels=["On time", "Late"], autopct="%1.2f%%",
        colors=[PALETTE["ontime"], PALETTE["late"]], startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Delivery outcome (share)")

    fig.suptitle("Class imbalance: late deliveries are the rare, expensive event",
                 fontsize=13, fontweight="bold")
    save(fig, "fig_eda_01_target_balance.png")

    note(f"Only {late_pct:.2f}% of delivered orders arrived late "
         f"({counts[1]:,} out of {len(df):,}). The dataset is imbalanced, so accuracy "
         f"alone is a misleading metric - always predicting 'on time' would score "
         f"{100 - late_pct:.1f}% while catching no late orders at all.")


# ----------------------------------------------------------------------------
# CHART 2  -  Does distance matter?  (our engineered star feature)
# ----------------------------------------------------------------------------
def chart_distance(df: pd.DataFrame) -> None:
    sub_banner("Chart 2: distance vs late rate")
    data = df.dropna(subset=["distance_km"]).copy()

    edges = [0, 50, 150, 400, 800, 1500, 4000]
    labels = ["0-50", "50-150", "150-400", "400-800", "800-1500", "1500 km +"]
    data["distance_band"] = pd.cut(data["distance_km"], bins=edges, labels=labels)

    grouped = data.groupby("distance_band", observed=True).agg(
        orders=("is_late", "size"), late_rate=("is_late", "mean")
    )
    grouped["late_rate"] *= 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    axes[0].hist(data.loc[data.is_late == 0, "distance_km"], bins=60, alpha=0.75,
                 label="On time", color=PALETTE["ontime"], density=True)
    axes[0].hist(data.loc[data.is_late == 1, "distance_km"], bins=60, alpha=0.65,
                 label="Late", color=PALETTE["late"], density=True)
    axes[0].set_xlabel("Seller to customer distance (km)")
    axes[0].set_ylabel("Share of orders")
    axes[0].set_title("Distance distribution by outcome")
    axes[0].legend()

    bars = axes[1].bar(grouped.index.astype(str), grouped["late_rate"],
                       color=PALETTE["late"], width=0.6)
    axes[1].axhline(df["is_late"].mean() * 100, color=PALETTE["grey"],
                    linestyle="--", label="Overall average")
    axes[1].set_xlabel("Distance band (km)")
    axes[1].set_ylabel("Late rate (%)")
    axes[1].set_title("Late rate rises with distance")
    axes[1].legend()
    for bar, value in zip(bars, grouped["late_rate"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Engineered feature: physical distance between seller and buyer",
                 fontsize=13, fontweight="bold")
    save(fig, "fig_eda_02_distance.png")

    lowest, highest = grouped["late_rate"].iloc[0], grouped["late_rate"].iloc[-1]
    note(f"Distance is strongly predictive. Orders travelling under 50 km are late "
         f"{lowest:.1f}% of the time, rising to {highest:.1f}% beyond 1,500 km - roughly "
         f"{highest / lowest:.1f} times higher. This column did not exist in the raw data; "
         f"we engineered it from the geolocation table.")


# ----------------------------------------------------------------------------
# CHART 3  -  Does the size of the promise matter?
# ----------------------------------------------------------------------------
def chart_promise_window(df: pd.DataFrame) -> None:
    sub_banner("Chart 3: promised delivery window vs late rate")
    data = df[df["estimated_delivery_days"].between(0, 60)].copy()

    edges = [0, 7, 14, 21, 30, 60]
    labels = ["0-7", "8-14", "15-21", "22-30", "31-60"]
    data["promise_band"] = pd.cut(data["estimated_delivery_days"], bins=edges, labels=labels)

    grouped = data.groupby("promise_band", observed=True).agg(
        orders=("is_late", "size"), late_rate=("is_late", "mean")
    )
    grouped["late_rate"] *= 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    sns.boxplot(data=data, x="is_late", y="estimated_delivery_days", ax=axes[0],
                palette=[PALETTE["ontime"], PALETTE["late"]], width=0.5)
    axes[0].set_xticklabels(["On time", "Late"])
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Promised window (days)")
    axes[0].set_title("Promised window by outcome")

    bars = axes[1].bar(grouped.index.astype(str), grouped["late_rate"],
                       color=PALETTE["accent"], width=0.6)
    axes[1].set_xlabel("Days promised at checkout")
    axes[1].set_ylabel("Late rate (%)")
    axes[1].set_title("Tight promises are broken far more often")
    for bar, value in zip(bars, grouped["late_rate"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("The delivery promise made at checkout", fontsize=13, fontweight="bold")
    save(fig, "fig_eda_03_promise_window.png")

    note(f"The promised window behaves as expected: orders promised within 7 days are "
         f"late {grouped['late_rate'].iloc[0]:.1f}% of the time, versus "
         f"{grouped['late_rate'].iloc[-1]:.1f}% for orders given 31-60 days. "
         f"This feature is legitimate because the promised date is displayed to the "
         f"customer at checkout, so it is genuinely known at prediction time.")


# ----------------------------------------------------------------------------
# CHART 4  -  What happened over time?
# ----------------------------------------------------------------------------
def chart_monthly_trend(df: pd.DataFrame) -> None:
    sub_banner("Chart 4: monthly trend")
    data = df.copy()
    data["month"] = data["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()

    monthly = data.groupby("month").agg(
        orders=("is_late", "size"), late_rate=("is_late", "mean")
    )
    monthly = monthly[monthly["orders"] >= 100]
    monthly["late_rate"] *= 100

    fig, ax1 = plt.subplots(figsize=(12, 4.6))
    ax1.bar(monthly.index, monthly["orders"], width=20,
            color=PALETTE["ontime"], alpha=0.65, label="Orders")
    ax1.set_ylabel("Orders per month", color=PALETTE["ontime"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["ontime"])

    ax2 = ax1.twinx()
    ax2.plot(monthly.index, monthly["late_rate"], color=PALETTE["late"],
             marker="o", linewidth=2.2, label="Late rate")
    ax2.set_ylabel("Late rate (%)", color=PALETTE["late"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["late"])
    ax2.grid(False)

    worst = monthly["late_rate"].idxmax()
    ax2.annotate(
        f"Peak: {monthly['late_rate'].max():.1f}%\n{worst:%b %Y}",
        xy=(worst, monthly["late_rate"].max()),
        xytext=(10, 18), textcoords="offset points",
        fontweight="bold", color=PALETTE["late"],
        arrowprops={"arrowstyle": "->", "color": PALETTE["late"]},
    )

    ax1.set_title("Order volume and late-delivery rate over time", fontweight="bold")
    save(fig, "fig_eda_04_monthly_trend.png")

    best = monthly["late_rate"].idxmin()
    nov_2017 = monthly["late_rate"].get(pd.Timestamp("2017-11-01"), float("nan"))
    note(f"Late deliveries are highly unstable over time, swinging from "
         f"{monthly['late_rate'].min():.1f}% in {best:%B %Y} to "
         f"{monthly['late_rate'].max():.1f}% in {worst:%B %Y} - a "
         f"{monthly['late_rate'].max() / monthly['late_rate'].min():.0f}-fold difference "
         f"around an overall average of {df['is_late'].mean() * 100:.1f}%. Two distinct "
         f"disruptions are visible: a sharp Black Friday spike in November 2017 "
         f"({nov_2017:.1f}%, roughly triple the preceding months) and a sustained "
         f"deterioration through February-March 2018 that the dataset alone cannot explain. "
         f"Because performance depends so heavily on when an order is placed, purchase "
         f"month and week are worth giving to the model.")


# ----------------------------------------------------------------------------
# CHART 5  -  Which parts of Brazil struggle?
# ----------------------------------------------------------------------------
def chart_state_analysis(df: pd.DataFrame) -> None:
    sub_banner("Chart 5: geography")
    grouped = df.groupby("customer_state").agg(
        orders=("is_late", "size"), late_rate=("is_late", "mean"),
        avg_distance=("distance_km", "mean"),
    )
    grouped = grouped[grouped["orders"] >= 200].sort_values("late_rate", ascending=False)
    grouped["late_rate"] *= 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colours = [PALETTE["late"] if v > df["is_late"].mean() * 100 else PALETTE["ontime"]
               for v in grouped["late_rate"]]
    axes[0].barh(grouped.index[::-1], grouped["late_rate"][::-1], color=colours[::-1])
    axes[0].axvline(df["is_late"].mean() * 100, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Late rate (%)")
    axes[0].set_title("Late rate by destination state")

    axes[1].scatter(grouped["avg_distance"], grouped["late_rate"],
                    s=grouped["orders"] / 40, alpha=0.65, color=PALETTE["late"],
                    edgecolor="black", linewidth=0.5)
    for state, row in grouped.iterrows():
        axes[1].annotate(state, (row["avg_distance"], row["late_rate"]),
                         fontsize=7.5, alpha=0.85)
    axes[1].set_xlabel("Average distance from seller (km)")
    axes[1].set_ylabel("Late rate (%)")
    axes[1].set_title("Remote states are hit hardest\n(bubble size = order volume)")

    fig.suptitle("Geographic inequality in delivery performance",
                 fontsize=13, fontweight="bold")
    save(fig, "fig_eda_05_state_analysis.png")

    worst = grouped.index[0]
    best = grouped.index[-1]
    note(f"Delivery reliability varies sharply by region. {worst} has the highest late rate "
         f"at {grouped.loc[worst, 'late_rate']:.1f}%, while {best} is the most reliable at "
         f"{grouped.loc[best, 'late_rate']:.1f}%. States far from the Sao Paulo seller hub "
         f"perform consistently worse.")


# ----------------------------------------------------------------------------
# CHART 6  -  Which products are risky?
# ----------------------------------------------------------------------------
def chart_category_analysis(df: pd.DataFrame) -> None:
    sub_banner("Chart 6: product categories")
    grouped = df.groupby("primary_category").agg(
        orders=("is_late", "size"), late_rate=("is_late", "mean"),
        avg_weight=("total_weight_g", "mean"),
    )
    grouped = grouped[grouped["orders"] >= 300].sort_values("late_rate", ascending=False)
    grouped["late_rate"] *= 100
    top = pd.concat([grouped.head(10), grouped.tail(5)])

    fig, ax = plt.subplots(figsize=(10, 6))
    colours = [PALETTE["late"]] * 10 + [PALETTE["ontime"]] * len(grouped.tail(5))
    ax.barh(top.index[::-1], top["late_rate"][::-1], color=colours[::-1])
    ax.axvline(df["is_late"].mean() * 100, color="black", linestyle="--", linewidth=1,
               label="Overall average")
    ax.set_xlabel("Late rate (%)")
    ax.set_title("Riskiest and safest product categories\n(categories with 300+ orders)",
                 fontweight="bold")
    ax.legend()
    save(fig, "fig_eda_06_category_analysis.png")

    note(f"Product category matters: '{grouped.index[0]}' is late "
         f"{grouped['late_rate'].iloc[0]:.1f}% of the time versus only "
         f"{grouped['late_rate'].iloc[-1]:.1f}% for '{grouped.index[-1]}'. Bulky and "
         f"fragile categories are over-represented at the risky end.")


# ----------------------------------------------------------------------------
# CHART 7  -  Distributions and outliers
# ----------------------------------------------------------------------------
def chart_distributions(df: pd.DataFrame) -> None:
    sub_banner("Chart 7: numeric distributions")
    columns = [
        ("total_price", "Order value (R$)"),
        ("total_freight", "Freight paid (R$)"),
        ("total_weight_g", "Total weight (g)"),
        ("distance_km", "Distance (km)"),
        ("estimated_delivery_days", "Promised window (days)"),
        ("actual_delivery_days", "Actual delivery (days)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (col, label) in zip(axes.ravel(), columns):
        series = df[col].dropna()
        upper = series.quantile(0.99)          # trim the extreme 1% so the shape is visible
        ax.hist(series[series <= upper], bins=50, color=PALETTE["ontime"], alpha=0.85)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Orders")
        ax.axvline(series.median(), color=PALETTE["late"], linestyle="--",
                   label=f"median {series.median():,.0f}")
        ax.legend(fontsize=8)

    fig.suptitle("Distribution of the main numeric features (top 1% trimmed for readability)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_eda_07_distributions.png")

    sub_banner("Chart 8: outliers")
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, col in zip(axes, ["total_price", "total_freight", "total_weight_g", "distance_km"]):
        ax.boxplot(df[col].dropna(), vert=True, widths=0.5,
                   flierprops={"markersize": 2, "alpha": 0.3})
        ax.set_title(col, fontsize=10)
        ax.set_xticks([])
    fig.suptitle("Outlier check: every money and size column has a long right tail",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_eda_08_outliers.png")

    price = df["total_price"]
    q1, q3 = price.quantile([0.25, 0.75])
    iqr = q3 - q1
    n_outliers = ((price < q1 - 1.5 * iqr) | (price > q3 + 1.5 * iqr)).sum()
    note(f"All money and size columns are heavily right-skewed. Order value has a median of "
         f"R$ {price.median():,.0f} but a maximum of R$ {price.max():,.0f}; "
         f"{n_outliers:,} orders ({n_outliers / len(df) * 100:.1f}%) fall outside the standard "
         f"1.5x IQR range. We keep these rather than delete them - they are genuine "
         f"high-value purchases, and tree-based models are not disturbed by skew.")


# ----------------------------------------------------------------------------
# CHART 9  -  Correlation
# ----------------------------------------------------------------------------
def chart_correlation(df: pd.DataFrame) -> None:
    sub_banner("Chart 9: correlation heatmap")
    columns = [
        "is_late", "distance_km", "estimated_delivery_days", "total_price",
        "total_freight", "total_weight_g", "n_items", "n_sellers",
        "max_installments", "freight_ratio", "same_state", "purchase_month",
    ]
    corr = df[columns].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(9.5, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.75}, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Correlation between numeric features and the target",
                 fontweight="bold", pad=14)
    save(fig, "fig_eda_09_correlation.png")

    target_corr = corr["is_late"].drop("is_late").sort_values(key=abs, ascending=False)
    top3 = ", ".join(f"{name} ({value:+.3f})" for name, value in target_corr.head(3).items())
    note(f"No single numeric column correlates strongly with lateness on its own. The "
         f"strongest linear relationships are {top3}. Weak individual correlations are "
         f"precisely why a linear model will struggle here and why tree-based ensembles, "
         f"which capture interactions between features, should perform better.")


# ----------------------------------------------------------------------------
# CHART 10  -  The business case: late delivery destroys reviews
# ----------------------------------------------------------------------------
def chart_review_impact(df: pd.DataFrame) -> None:
    sub_banner("Chart 10: business impact on customer reviews")
    reviews = load_raw("reviews")[["order_id", "review_score"]].drop_duplicates("order_id")
    merged = df.merge(reviews, on="order_id", how="inner")

    # Percentage of each outcome group that gave each star rating.
    # normalize="index" makes every row sum to 100%.
    pivot = (
        pd.crosstab(merged["is_late"], merged["review_score"], normalize="index")
        .mul(100)
        .stack()
        .rename("pct")
        .reset_index()
    )
    pivot["is_late"] = pivot["is_late"].map({0: "On time", 1: "Late"})

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    sns.barplot(data=pivot, x="review_score", y="pct", hue="is_late", ax=axes[0],
                palette=[PALETTE["ontime"], PALETTE["late"]],
                hue_order=["On time", "Late"])
    axes[0].set_xlabel("Customer review score (1 = worst, 5 = best)")
    axes[0].set_ylabel("Share of orders (%)")
    axes[0].set_title("Review scores collapse when delivery is late")
    axes[0].legend(title="")

    avg = merged.groupby("is_late")["review_score"].mean()
    bad = merged.assign(bad=merged["review_score"] <= 2).groupby("is_late")["bad"].mean() * 100
    bars = axes[1].bar(["On time", "Late"], bad.values,
                       color=[PALETTE["ontime"], PALETTE["late"]], width=0.5)
    axes[1].set_ylabel("Orders rated 1 or 2 stars (%)")
    axes[1].set_title("Share of very poor reviews")
    for bar, value in zip(bars, bad.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%",
                     ha="center", va="bottom", fontweight="bold")

    fig.suptitle("Why this matters commercially", fontsize=13, fontweight="bold")
    save(fig, "fig_eda_10_review_impact.png")

    note(f"This is the commercial justification for the whole project. Orders delivered on "
         f"time average {avg[0]:.2f} stars; late orders average {avg[1]:.2f}. The share of "
         f"1-2 star reviews jumps from {bad[0]:.1f}% to {bad[1]:.1f}% - a "
         f"{bad[1] / bad[0]:.1f}x increase. Every late delivery predicted in advance is a "
         f"chance to protect a customer relationship before it is damaged.")


# ----------------------------------------------------------------------------
# CHART 11  -  Missing values
# ----------------------------------------------------------------------------
def chart_missing(df: pd.DataFrame) -> None:
    sub_banner("Chart 11: missing values in the assembled table")
    missing = (df.isna().sum() / len(df) * 100)
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("    no missing values - chart skipped")
        return

    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.36 * len(missing))))
    ax.barh(missing.index[::-1], missing.values[::-1], color=PALETTE["accent"])
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing values remaining after the nine-table join", fontweight="bold")
    for i, value in enumerate(missing.values[::-1]):
        ax.text(value, i, f" {value:.2f}%", va="center", fontsize=8.5)
    save(fig, "fig_eda_11_missing_values.png")

    note(f"After joining, {len(missing)} columns still contain gaps. The largest is "
         f"'{missing.index[0]}' at {missing.iloc[0]:.2f}%. All of these are handled inside "
         f"the modelling pipeline by imputation, never by deleting rows.")


def main() -> None:
    configure_pandas()

    banner("OLIST DELIVERY-DELAY PREDICTION  |  SCRIPT 03: EXPLORATORY DATA ANALYSIS")

    path = PROCESSED_DIR / "olist_analytical.parquet"
    if not path.exists():
        print(f"  ERROR: {path} not found. Run 02_build_dataset.py first.")
        return 1

    df = pd.read_parquet(path)
    print(f"  Loaded analytical table: {df.shape[0]:,} orders x {df.shape[1]} columns")
    print(f"  Charts will be saved to: {FIGURES_DIR}")

    banner("GENERATING CHARTS")
    chart_target_balance(df)
    chart_distance(df)
    chart_promise_window(df)
    chart_monthly_trend(df)
    chart_state_analysis(df)
    chart_category_analysis(df)
    chart_distributions(df)
    chart_correlation(df)
    chart_review_impact(df)
    chart_missing(df)

    banner("KEY OBSERVATIONS  -  Task 1.1 deliverable")
    for i, observation in enumerate(OBSERVATIONS, start=1):
        print(f"\n  [{i}] {observation}")

    with open(FIGURES_DIR.parent / "key_observations.txt", "w", encoding="utf-8") as fh:
        for i, observation in enumerate(OBSERVATIONS, start=1):
            fh.write(f"[{i}] {observation}\n\n")

    banner("EDA COMPLETE")
    print(f"  {len(list(FIGURES_DIR.glob('*.png')))} charts written to reports/figures/")
    print(f"  Observations saved to reports/key_observations.txt")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
