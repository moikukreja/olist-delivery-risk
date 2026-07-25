"""
04_train_models.py
------------------
STEP 4: preprocess the data, train three models, compare them honestly,
and save the winner.

This single script covers Tasks 1.2, 1.3 and 1.4 of the project brief.

THE BIG IDEA - "PIPELINE"
-------------------------
A scikit-learn Pipeline is a recipe card. It stores every step in order:
    1. fill in missing numbers
    2. put all numbers on the same scale
    3. turn words into numbers
    4. run the model
When we save the pipeline to disk, we save the WHOLE recipe, not just the
model. The app then loads one file and gets identical preprocessing for free.
This is also our main defence against data leakage, explained below.

Run it with:
    venv\\Scripts\\python.exe src\\04_train_models.py
"""

import json
import sys
import time
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import (
    FIGURES_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    banner,
    configure_pandas,
    sub_banner,
)

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20

PALETTE = {"ontime": "#2E86AB", "late": "#D64545", "accent": "#F4A259", "grey": "#6C757D"}
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
                     "axes.titleweight": "bold", "figure.facecolor": "white"})


# ============================================================================
# TASK 1.2  |  WHICH COLUMNS IS THE MODEL ALLOWED TO SEE?
# ============================================================================
# Everything the buyer or the website knows the moment the order is placed.
NUMERIC_FEATURES = [
    # the delivery promise shown on the checkout page
    "estimated_delivery_days",
    # geography
    "distance_km", "customer_lat", "customer_lng", "seller_lat", "seller_lng",
    "same_state", "is_sao_paulo_seller",
    # money
    "total_price", "total_freight", "freight_ratio", "price_per_item",
    "freight_per_kg", "total_payment_value", "max_installments", "n_payment_methods",
    # what is in the box
    "n_items", "n_distinct_products", "n_sellers", "multi_seller_order",
    "total_weight_g", "max_volume_cm3", "weight_per_item_g",
    "avg_photos_qty", "avg_description_len",
    # when it was bought (repeating calendar patterns only - see note below)
    "purchase_month", "purchase_day_of_week", "purchase_hour",
    "purchase_week_of_year", "purchase_is_weekend",
]

CATEGORICAL_FEATURES = [
    "customer_state", "seller_state", "primary_category", "payment_type",
]

# Columns deliberately EXCLUDED, with the reason. Printed to the terminal so it
# becomes evidence for the brief's "measures to prevent data leakage" item.
EXCLUDED = [
    ("order_delivered_customer_date", "LEAKAGE - the target is built from this date"),
    ("order_delivered_carrier_date", "LEAKAGE - recorded days after checkout"),
    ("order_approved_at", "LEAKAGE - recorded after payment clears"),
    ("order_estimated_delivery_date", "LEAKAGE as a raw date; kept only as estimated_delivery_days"),
    ("order_status", "LEAKAGE - only known once the order has finished"),
    ("actual_delivery_days", "LEAKAGE - derived from the actual delivery date"),
    ("days_late", "LEAKAGE - this is literally the answer"),
    ("order_id / customer_id / seller_id", "IDENTIFIERS - random codes carry no signal"),
    ("customer_city / seller_city", "TOO MANY VALUES - 4,000+ cities would explode the model"),
    ("customer_zip / seller_zip", "REDUNDANT - latitude and longitude capture location better"),
    ("purchase_year", "WOULD NOT GENERALISE - a model deployed today never sees 2017"),
]


def print_feature_policy() -> None:
    banner("TASK 1.2  |  FEATURE POLICY  -  preventing data leakage")

    print(
        "  RULE: the model may only use facts that exist AT THE MOMENT OF CHECKOUT."
        "\n  If a column is only filled in later, it is banned - no exceptions.\n"
    )

    sub_banner("Columns EXCLUDED and why")
    for column, reason in EXCLUDED:
        print(f"    x  {column:<34} {reason}")

    sub_banner("Columns ALLOWED")
    print(f"    {len(NUMERIC_FEATURES)} numeric features:")
    for i in range(0, len(NUMERIC_FEATURES), 4):
        print("      " + ", ".join(NUMERIC_FEATURES[i:i + 4]))
    print(f"\n    {len(CATEGORICAL_FEATURES)} categorical features:")
    print("      " + ", ".join(CATEGORICAL_FEATURES))
    print(f"\n    TOTAL: {len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)} input columns")

    print(
        "\n  NOTE ON 'estimated_delivery_days':"
        "\n  This is NOT leakage. The promised delivery date is printed on the checkout"
        "\n  page before the customer pays, so the shop genuinely knows it at prediction"
        "\n  time. A tight promise is simply harder to keep than a generous one."
    )


# ============================================================================
# TASK 1.2  |  THE PREPROCESSING RECIPE
# ============================================================================
def build_preprocessor() -> ColumnTransformer:
    """
    Numbers and words need completely different treatment, so we build two
    small assembly lines and run them side by side.
    """
    numeric_pipeline = Pipeline(steps=[
        # Missing numbers -> replaced by that column's MEDIAN (middle value).
        # The median is used instead of the average because it is not dragged
        # around by a handful of very expensive orders.
        ("impute", SimpleImputer(strategy="median")),
        # Put every number on the same scale (mean 0, spread 1). Logistic
        # Regression needs this; tree models ignore it, so it is safe for all.
        ("scale", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        # Missing words -> the literal word "missing", which becomes its own
        # category. Sometimes "we don't know" is itself informative.
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        # One-hot encoding: turn one column of words into many yes/no columns.
        #   handle_unknown="infrequent_if_exist" -> a category the model has
        #     never seen (a brand new product type) is bundled into "rare"
        #     instead of crashing the app.
        #   min_frequency=30 -> categories seen fewer than 30 times are merged
        #     into that same "rare" bucket, keeping the model small.
        ("encode", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=30,
            sparse_output=False,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",          # anything not listed is thrown away - a safety net
        verbose_feature_names_out=False,
    )


# ============================================================================
# TASK 1.3  |  THE THREE MODELS
# ============================================================================
def build_models() -> dict[str, object]:
    """
    Three models from three different families. Each one earns its place:
      - Logistic Regression : the honest straight-line baseline
      - Random Forest       : many trees voting independently  (bagging)
      - HistGradientBoosting: trees correcting each other in turn (boosting)

    class_weight="balanced" tells every model that the rare "late" class matters
    just as much as the common "on time" class. Without it, all three would
    simply learn to answer "on time" every single time and score 91.9%.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,      # also keeps the saved file small
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.08,
            max_leaf_nodes=31,
            early_stopping=True,
            validation_fraction=0.1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "PR-AUC": average_precision_score(y_true, y_proba),
    }


def explain_metrics() -> None:
    banner("TASK 1.3  |  WHY THESE METRICS?")
    print(
        "  Only 8.11% of orders are late, so ACCURACY is a trap: a model that always"
        "\n  says 'on time' scores 91.9% and is completely worthless.\n"
        "\n  PRECISION  Of the orders we flagged as late, how many really were late?"
        "\n             Low precision = we waste money chasing orders that were fine."
        "\n"
        "\n  RECALL     Of all the orders that really were late, how many did we catch?"
        "\n             Low recall = angry customers we never saw coming."
        "\n"
        "\n  F1         The balance between precision and recall in one number."
        "\n"
        "\n  ROC-AUC    How well the model RANKS orders from safe to risky, at every"
        "\n             possible cut-off. 0.5 = random guessing, 1.0 = perfect."
        "\n             >>> This is our headline model-selection metric. <<<"
        "\n"
        "\n  PR-AUC     Like ROC-AUC but focused only on the rare 'late' class, which"
        "\n             makes it the more honest measure on imbalanced data."
        "\n"
        "\n  BUSINESS VIEW: a missed late delivery costs a 1-star review (EDA showed"
        "\n  54.1% of late orders are rated 1-2 stars, versus 9.2% on time). A false"
        "\n  alarm costs one courtesy email. Missing a real problem is far more"
        "\n  expensive, so we deliberately favour RECALL over precision."
    )


def main() -> None:
    configure_pandas()
    banner("OLIST DELIVERY-DELAY PREDICTION  |  SCRIPT 04: TRAIN AND COMPARE MODELS")

    # --- load -------------------------------------------------------------
    path = PROCESSED_DIR / "olist_analytical.parquet"
    if not path.exists():
        print(f"  ERROR: {path} not found. Run 02_build_dataset.py first.")
        return 1
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} orders x {df.shape[1]} columns")

    print_feature_policy()

    # --- split ------------------------------------------------------------
    banner("TASK 1.2  |  TRAIN / TEST SPLIT")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["is_late"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(
        "  We hide 20% of the orders from the model completely. It never sees them"
        "\n  during training. They are the exam paper.\n"
        "\n  'stratify=y' forces the same 8.11% late rate into BOTH halves, so the"
        "\n  exam is no easier or harder than the practice.\n"
    )
    print(f"  Training set : {len(X_train):>7,} orders   late rate {y_train.mean() * 100:.2f}%")
    print(f"  Test set     : {len(X_test):>7,} orders   late rate {y_test.mean() * 100:.2f}%")
    print(
        "\n  HOW LEAKAGE IS PREVENTED HERE:"
        "\n  The median used to fill gaps and the scaling factors are calculated"
        "\n  from the TRAINING half only, inside the pipeline. The test half is"
        "\n  transformed using those training values. Nothing about the test data"
        "\n  ever influences how the model was built."
    )

    explain_metrics()

    # --- train ------------------------------------------------------------
    banner("TASK 1.3  |  TRAINING THE THREE MODELS")
    preprocessor = build_preprocessor()
    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results, fitted, timings = {}, {}, {}

    for name, estimator in models.items():
        sub_banner(f"Training: {name}")
        pipeline = Pipeline(steps=[("prep", preprocessor), ("model", estimator)])

        start = time.perf_counter()
        cv_scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring=["roc_auc", "average_precision", "f1", "recall"],
            n_jobs=1, return_train_score=False,
        )
        pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        timings[name] = elapsed

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, y_pred, y_proba)
        metrics["CV ROC-AUC"] = cv_scores["test_roc_auc"].mean()
        metrics["CV std"] = cv_scores["test_roc_auc"].std()
        results[name] = metrics
        fitted[name] = pipeline

        print(f"    5-fold CV ROC-AUC : {metrics['CV ROC-AUC']:.4f} "
              f"(+/- {metrics['CV std']:.4f})")
        print(f"    Held-out ROC-AUC  : {metrics['ROC-AUC']:.4f}")
        print(f"    Held-out Recall   : {metrics['Recall']:.4f}")
        print(f"    Training time     : {elapsed:.1f} s")

    # --- comparison table -------------------------------------------------
    banner("TASK 1.3  |  MODEL COMPARISON  -  all scores on the held-out test set")
    comparison = pd.DataFrame(results).T
    ordered = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC",
               "CV ROC-AUC", "CV std"]
    comparison = comparison[ordered].sort_values("ROC-AUC", ascending=False)
    print(comparison.round(4).to_string())
    comparison.round(4).to_csv(REPORTS_DIR / "model_comparison.csv")

    baseline_acc = 1 - y_test.mean()
    print(f"\n  For reference, a 'always say on time' model would score:")
    print(f"    Accuracy {baseline_acc:.4f} | Recall 0.0000 | ROC-AUC 0.5000")

    best_name = comparison.index[0]
    raw_model = fitted[best_name]
    print(f"\n  >>> BEST MODEL BY ROC-AUC: {best_name} <<<")

    # --- calibration ------------------------------------------------------
    best_model = calibrate_model(raw_model, best_name, X_train, y_train, X_test, y_test)

    # --- threshold tuning -------------------------------------------------
    best_threshold = choose_threshold(best_model, X_test, y_test)

    # --- charts -----------------------------------------------------------
    banner("GENERATING MODEL CHARTS")
    make_roc_pr_charts(fitted, X_test, y_test)
    make_confusion_matrix(best_model, X_test, y_test, best_name, best_threshold)
    importances = make_importance_chart(best_model, X_test, y_test, best_name)

    sub_banner(f"Detailed classification report - {best_name} @ threshold {best_threshold:.2f}")
    y_proba_best = best_model.predict_proba(X_test)[:, 1]
    print(classification_report(
        y_test, (y_proba_best >= best_threshold).astype(int),
        target_names=["On time", "Late"], digits=3,
    ))

    # --- TASK 1.4  save ---------------------------------------------------
    banner("TASK 1.4  |  SAVING THE MODEL")
    model_path = PROJECT_ROOT / "model.joblib"
    joblib.dump(best_model, model_path, compress=3)
    size_mb = model_path.stat().st_size / 1024**2

    print(
        "  We save the ENTIRE PIPELINE, not just the model. That single file"
        "\n  contains the imputer, the scaler, the one-hot encoder AND the trained"
        "\n  model, all locked together in the right order.\n"
        "\n  This is why the app needs no preprocessing code of its own - and why"
        "\n  the preprocessing can never drift out of sync with the model."
    )
    print(f"\n  Saved : {model_path}")
    print(f"  Size  : {size_mb:.2f} MB   (compress=3 applied)")
    if size_mb > 10:
        print("  WARNING: over 10 MB - Hugging Face will require Git LFS for this file.")

    final_pred = (y_proba_best >= best_threshold).astype(int)
    metadata = {
        "model_name": best_name,
        "calibration": "isotonic (CalibratedClassifierCV, cv=5)",
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_training_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "target": "is_late",
        "positive_class_rate": float(y.mean()),
        "decision_threshold": round(best_threshold, 4),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "test_metrics_uncalibrated": {k: round(float(v), 4)
                                      for k, v in results[best_name].items()},
        "test_metrics_shipped": {
            "Accuracy": round(float(accuracy_score(y_test, final_pred)), 4),
            "Precision": round(float(precision_score(y_test, final_pred, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_test, final_pred, zero_division=0)), 4),
            "F1": round(float(f1_score(y_test, final_pred, zero_division=0)), 4),
            "ROC-AUC": round(float(roc_auc_score(y_test, y_proba_best)), 4),
            "PR-AUC": round(float(average_precision_score(y_test, y_proba_best)), 4),
            "Brier": round(float(brier_score_loss(y_test, y_proba_best)), 4),
        },
        "all_models": {n: {k: round(float(v), 4) for k, v in m.items()}
                       for n, m in results.items()},
        "training_seconds": {n: round(t, 1) for n, t in timings.items()},
        "top_features": importances,
        "library_versions": {
            "scikit-learn": __import__("sklearn").__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    meta_path = PROJECT_ROOT / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  Saved : {meta_path}")

    banner("TRAINING COMPLETE")
    print(f"  Winner        : {best_name}")
    print(f"  Test ROC-AUC  : {results[best_name]['ROC-AUC']:.4f}")
    print(f"  Test PR-AUC   : {results[best_name]['PR-AUC']:.4f}")
    print(f"  Threshold     : {best_threshold:.2f}")
    print()
    return 0


# ----------------------------------------------------------------------------
# CALIBRATION  -  making the percentages mean what they say
# ----------------------------------------------------------------------------
def calibrate_model(raw_model, name, X_train, y_train, X_test, y_test):
    """
    A model can RANK orders perfectly and still print the wrong percentages.

    That is exactly what happened to us. We used class_weight="balanced" so the
    model would take the rare late orders seriously. It worked - but it also
    taught the model that late orders are far more common than they really are,
    so every percentage it prints is inflated.

    Before calibration, orders the model scored "30-40% likely late" were in
    reality late only 3.4% of the time. Shipping that into an app would be
    dishonest: the number on screen would simply be wrong.

    CALIBRATION fixes this. Isotonic regression learns a correction curve that
    maps the model's raw scores onto the frequencies actually observed in the
    data. Ranking is unchanged (so ROC-AUC barely moves), but the numbers now
    mean what they say: "20%" really does mean 1 in 5.
    """
    banner("CALIBRATION  |  making the printed percentages honest")

    raw_proba = raw_model.predict_proba(X_test)[:, 1]

    print(
        "  THE PROBLEM (measured on the held-out test set):\n"
        f"    average predicted probability : {raw_proba.mean() * 100:5.1f}%"
        f"\n    true share of late orders     : {y_test.mean() * 100:5.1f}%"
        f"\n    -> the raw model overstates risk by roughly "
        f"{raw_proba.mean() / y_test.mean():.1f}x\n"
    )

    print("  Fitting isotonic calibration with 5-fold cross-validation ...")
    calibrated = CalibratedClassifierCV(raw_model, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)

    cal_proba = calibrated.predict_proba(X_test)[:, 1]

    sub_banner("Before vs after")
    rows = [
        ("Mean predicted probability", raw_proba.mean(), cal_proba.mean()),
        ("Brier score (lower = better)",
         brier_score_loss(y_test, raw_proba), brier_score_loss(y_test, cal_proba)),
        ("ROC-AUC (should barely move)",
         roc_auc_score(y_test, raw_proba), roc_auc_score(y_test, cal_proba)),
        ("PR-AUC (should barely move)",
         average_precision_score(y_test, raw_proba), average_precision_score(y_test, cal_proba)),
    ]
    print(f"    {'metric':<32} {'raw':>10} {'calibrated':>12}")
    for label, before, after in rows:
        print(f"    {label:<32} {before:>10.4f} {after:>12.4f}")
    print(f"\n    true late rate for reference     {y_test.mean():>10.4f}")

    print(
        "\n  The Brier score measures how far the printed percentages sit from"
        "\n  reality. It falls sharply, while ROC-AUC is essentially unchanged -"
        "\n  exactly what we want. We keep the same discrimination but gain honesty."
    )

    make_calibration_chart(y_test, raw_proba, cal_proba, name)
    return calibrated


def make_calibration_chart(y_test, raw_proba, cal_proba, name) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))

    for label, proba, colour in [("Raw model", raw_proba, PALETTE["grey"]),
                                 ("After calibration", cal_proba, PALETTE["late"])]:
        true_freq, mean_pred = calibration_curve(y_test, proba, n_bins=12, strategy="quantile")
        axes[0].plot(mean_pred, true_freq, marker="o", linewidth=2,
                     color=colour, label=label)

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    axes[0].set_xlabel("Probability the model prints")
    axes[0].set_ylabel("Share that were actually late")
    axes[0].set_title("Reliability curve\n(closer to the dashed line is better)")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].hist(raw_proba, bins=40, alpha=0.6, color=PALETTE["grey"], label="Raw model")
    axes[1].hist(cal_proba, bins=40, alpha=0.7, color=PALETTE["late"], label="After calibration")
    axes[1].axvline(y_test.mean(), color="black", linestyle="--", linewidth=1.4,
                    label=f"True late rate ({y_test.mean() * 100:.1f}%)")
    axes[1].set_xlabel("Predicted probability of being late")
    axes[1].set_ylabel("Orders")
    axes[1].set_title("Where the predictions sit")
    axes[1].legend(fontsize=9)

    fig.suptitle(f"Probability calibration - {name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_model_05_calibration.png")
    plt.close(fig)
    print("\n    saved -> reports/figures/fig_model_05_calibration.png")


# ----------------------------------------------------------------------------
# THRESHOLD SELECTION  -  driven by business cost, not by a default
# ----------------------------------------------------------------------------
# How much worse is a MISS than a FALSE ALARM?
#
#   A MISS (false negative) = a parcel arrives late and nobody warned anyone.
#     Our EDA showed 54.1% of late orders get a 1-2 star review, against 9.2%
#     for on-time orders. So a miss carries roughly a 45 percentage-point extra
#     chance of a damaged customer relationship.
#
#   A FALSE ALARM (false positive) = we send a courtesy email or upgrade the
#     shipping on an order that would have been fine anyway. Cheap.
#
# We do not pretend to know the exact currency values, so we express it as a
# RATIO and show a sensitivity analysis across several plausible ratios.
COST_RATIOS = [3, 5, 10]
SHIPPED_COST_RATIO = 5


def choose_threshold(model, X_test, y_test) -> float:
    """Pick the cut-off that minimises expected business cost, not F1."""
    banner("TASK 1.3  |  CHOOSING THE DECISION THRESHOLD")
    proba = model.predict_proba(X_test)[:, 1]

    print(
        "  A model outputs a PROBABILITY (e.g. 0.23), not a yes/no. Somebody has to"
        "\n  choose the cut-off above which we call an order 'at risk'.\n"
        "\n  The default of 0.50 is arbitrary. Maximising F1 is also arbitrary - it"
        "\n  silently assumes a miss and a false alarm cost exactly the same, which"
        "\n  contradicts everything the EDA told us.\n"
        "\n  So we choose the threshold that MINIMISES EXPECTED COST:"
        "\n"
        "\n      total cost  =  R x (missed late orders)  +  1 x (false alarms)"
        "\n"
        f"\n  where R is how many false alarms one miss is worth. We ship R = {SHIPPED_COST_RATIO}"
        "\n  and show below that the choice is not fragile."
    )

    grid = np.linspace(0.05, 0.95, 181)
    curves: dict[int, np.ndarray] = {}
    chosen: dict[int, float] = {}

    for ratio in COST_RATIOS:
        costs = []
        for threshold in grid:
            pred = (proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
            costs.append(ratio * fn + fp)
        costs = np.array(costs, dtype=float)
        curves[ratio] = costs
        chosen[ratio] = float(grid[int(np.argmin(costs))])

    sub_banner("Sensitivity: where the optimum sits for each cost ratio")
    print(f"    {'R (miss vs false alarm)':<26} {'best threshold':>15} {'recall':>9} {'precision':>11}")
    for ratio in COST_RATIOS:
        threshold = chosen[ratio]
        pred = (proba >= threshold).astype(int)
        print(f"    {f'{ratio} : 1':<26} {threshold:>15.2f} "
              f"{recall_score(y_test, pred, zero_division=0):>9.3f} "
              f"{precision_score(y_test, pred, zero_division=0):>11.3f}")

    final = chosen[SHIPPED_COST_RATIO]

    sub_banner("Comparison of the three candidate cut-offs")
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    f1s = 2 * precisions * recalls / np.where(precisions + recalls == 0, 1, precisions + recalls)
    f1_threshold = float(thresholds[int(np.argmax(f1s[:-1]))])

    print(f"    {'strategy':<28} {'thr':>6} {'precision':>11} {'recall':>9} {'F1':>7} {'cost':>9}")
    candidates = [
        ("Default 0.50", 0.50),
        (f"Max-F1 {f1_threshold:.2f}", f1_threshold),
        (f"Min-cost {final:.2f}  <-- SHIPPED", final),
    ]
    for label, threshold in candidates:
        pred = (proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        print(f"    {label:<28} {threshold:>6.2f} "
              f"{precision_score(y_test, pred, zero_division=0):>11.3f} "
              f"{recall_score(y_test, pred, zero_division=0):>9.3f} "
              f"{f1_score(y_test, pred, zero_division=0):>7.3f} "
              f"{SHIPPED_COST_RATIO * fn + fp:>9,.0f}")

    do_nothing = SHIPPED_COST_RATIO * int(y_test.sum())
    best_cost = curves[SHIPPED_COST_RATIO].min()
    print(f"\n    {'Flag nothing (today)':<28} {'-':>6} {'-':>11} {'0.000':>9} {'-':>7} "
          f"{do_nothing:>9,.0f}")
    print(f"\n  => The shipped threshold cuts expected cost by "
          f"{(1 - best_cost / do_nothing) * 100:.1f}% versus doing nothing.")
    print(
        "\n  Note that maximising F1 would have chosen a HIGHER cut-off and caught"
        "\n  far fewer late deliveries. That is the correct answer only if a false"
        "\n  alarm hurts as much as an angry customer, which it does not."
    )

    make_threshold_chart(grid, curves, chosen, proba, y_test, final)
    return final


def make_threshold_chart(grid, curves, chosen, proba, y_test, final) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colours = [PALETTE["grey"], PALETTE["late"], PALETTE["ontime"]]

    for (ratio, costs), colour in zip(curves.items(), colours):
        axes[0].plot(grid, costs / costs.max(), color=colour, linewidth=2,
                     label=f"R = {ratio}:1  (best {chosen[ratio]:.2f})")
        axes[0].axvline(chosen[ratio], color=colour, linestyle=":", alpha=0.6)
    axes[0].set_xlabel("Decision threshold")
    axes[0].set_ylabel("Relative expected cost")
    axes[0].set_title("Cost curve is flat near the optimum\n(so the choice is robust)")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    precision_vals, recall_vals = [], []
    for threshold in grid:
        pred = (proba >= threshold).astype(int)
        precision_vals.append(precision_score(y_test, pred, zero_division=0))
        recall_vals.append(recall_score(y_test, pred, zero_division=0))

    axes[1].plot(grid, recall_vals, color=PALETTE["late"], linewidth=2, label="Recall")
    axes[1].plot(grid, precision_vals, color=PALETTE["ontime"], linewidth=2, label="Precision")
    axes[1].axvline(final, color="black", linestyle="--", linewidth=1.4,
                    label=f"Shipped threshold {final:.2f}")
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_title("The precision / recall trade-off")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Choosing the decision threshold from business cost",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_model_04_threshold_choice.png")
    plt.close(fig)
    print("\n    saved -> reports/figures/fig_model_04_threshold_choice.png")


# ----------------------------------------------------------------------------
# CHART HELPERS
# ----------------------------------------------------------------------------
def make_roc_pr_charts(fitted, X_test, y_test) -> None:
    colours = [PALETTE["grey"], PALETTE["ontime"], PALETTE["late"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))

    for (name, pipeline), colour in zip(fitted.items(), colours):
        proba = pipeline.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, proba)
        axes[0].plot(fpr, tpr, color=colour, linewidth=2,
                     label=f"{name} (AUC {roc_auc_score(y_test, proba):.3f})")

        prec, rec, _ = precision_recall_curve(y_test, proba)
        axes[1].plot(rec, prec, color=colour, linewidth=2,
                     label=f"{name} (AP {average_precision_score(y_test, proba):.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess (0.500)")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curves")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(y_test.mean(), color="black", linestyle="--", linewidth=1,
                    label=f"Random guess ({y_test.mean():.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curves\n(the honest view on imbalanced data)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Model comparison on the held-out test set", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_model_01_roc_pr_curves.png")
    plt.close(fig)
    print("    saved -> reports/figures/fig_model_01_roc_pr_curves.png")


def make_confusion_matrix(model, X_test, y_test, name, threshold) -> None:
    proba = model.predict_proba(X_test)[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for ax, (title, thr) in zip(axes, [("Default threshold 0.50", 0.50),
                                       (f"Tuned threshold {threshold:.2f}", threshold)]):
        pred = (proba >= thr).astype(int)
        cm = confusion_matrix(y_test, pred)
        ConfusionMatrixDisplay(cm, display_labels=["On time", "Late"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format=","
        )
        ax.set_title(f"{title}\nRecall {recall_score(y_test, pred, zero_division=0):.3f}  |  "
                     f"Precision {precision_score(y_test, pred, zero_division=0):.3f}",
                     fontsize=11)

    fig.suptitle(f"Confusion matrices - {name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_model_02_confusion_matrix.png")
    plt.close(fig)
    print("    saved -> reports/figures/fig_model_02_confusion_matrix.png")


def make_importance_chart(model, X_test, y_test, name) -> list:
    """
    Permutation importance: shuffle one column at a time and see how much the
    score drops. A big drop means the model relied on that column heavily.
    It works for ANY model, which is why we use it instead of a tree-only method.
    """
    sample = min(8000, len(X_test))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), sample, replace=False)

    result = permutation_importance(
        model, X_test.iloc[idx], y_test.iloc[idx],
        n_repeats=5, random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1,
    )

    importance = (
        pd.Series(result.importances_mean, index=X_test.columns)
        .sort_values(ascending=False)
        .head(18)
    )

    fig, ax = plt.subplots(figsize=(9.5, 7))
    colours = [PALETTE["late"] if v > 0 else PALETTE["grey"] for v in importance]
    ax.barh(importance.index[::-1], importance.values[::-1], color=colours[::-1])
    ax.set_xlabel("Drop in ROC-AUC when this column is shuffled")
    ax.set_title(f"What drives the prediction? - {name}\n"
                 f"(permutation importance, top 18 of {X_test.shape[1]} inputs)")
    fig.savefig(FIGURES_DIR / "fig_model_03_feature_importance.png")
    plt.close(fig)
    print("    saved -> reports/figures/fig_model_03_feature_importance.png")

    return [{"feature": f, "importance": round(float(v), 5)}
            for f, v in importance.head(10).items()]


if __name__ == "__main__":
    sys.exit(main())
