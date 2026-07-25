"""
05_smoke_test.py
----------------
A "smoke test" is a quick check that the thing turns on without catching fire.

Before we build the app, we prove three things:
  1. model.joblib loads from disk without errors
  2. it accepts a single hand-typed order (exactly what the app will send it)
  3. the probabilities it returns are sensible - a risky order should score
     higher than a safe one

If this script passes, any later problem is in the app code, not the model.

Run it with:
    venv\\Scripts\\python.exe src\\05_smoke_test.py
"""

import json
import sys

import joblib
import pandas as pd

from utils import PROJECT_ROOT, banner, sub_banner

# Two deliberately contrasting orders, typed out by hand the way a user would
# fill in the app's form.
SAFE_ORDER = {
    "estimated_delivery_days": 30.0,   # generous promise
    "distance_km": 15.0,               # same city
    "customer_lat": -23.55, "customer_lng": -46.63,
    "seller_lat": -23.56, "seller_lng": -46.64,
    "same_state": 1, "is_sao_paulo_seller": 1,
    "total_price": 90.0, "total_freight": 12.0, "freight_ratio": 0.133,
    "price_per_item": 90.0, "freight_per_kg": 24.0,
    "total_payment_value": 102.0, "max_installments": 1, "n_payment_methods": 1,
    "n_items": 1, "n_distinct_products": 1, "n_sellers": 1, "multi_seller_order": 0,
    "total_weight_g": 500.0, "max_volume_cm3": 2000.0, "weight_per_item_g": 500.0,
    "avg_photos_qty": 3.0, "avg_description_len": 800.0,
    "purchase_month": 6, "purchase_day_of_week": 1, "purchase_hour": 14,
    "purchase_week_of_year": 24, "purchase_is_weekend": 0,
    "customer_state": "SP", "seller_state": "SP",
    "primary_category": "housewares", "payment_type": "credit_card",
}

# A small parcel sent right across the country on a tight promise, during the
# congested Black Friday week. This is the classic risky profile in this data.
RISKY_ORDER = {
    "estimated_delivery_days": 7.0,    # tight promise
    "distance_km": 2600.0,             # Sao Paulo -> Manaus, across the country
    "customer_lat": -3.10, "customer_lng": -60.02,   # Manaus, far north
    "seller_lat": -23.56, "seller_lng": -46.64,      # Sao Paulo
    "same_state": 0, "is_sao_paulo_seller": 1,
    "total_price": 480.0, "total_freight": 22.0, "freight_ratio": 0.046,
    "price_per_item": 480.0, "freight_per_kg": 11.0,
    "total_payment_value": 502.0, "max_installments": 8, "n_payment_methods": 1,
    "n_items": 1, "n_distinct_products": 1, "n_sellers": 1, "multi_seller_order": 0,
    "total_weight_g": 2000.0, "max_volume_cm3": 8000.0, "weight_per_item_g": 2000.0,
    "avg_photos_qty": 2.0, "avg_description_len": 400.0,
    "purchase_month": 11, "purchase_day_of_week": 4, "purchase_hour": 22,
    "purchase_week_of_year": 47, "purchase_is_weekend": 0,   # Black Friday week
    "customer_state": "AM", "seller_state": "SP",
    "primary_category": "furniture_decor", "payment_type": "credit_card",
}


def main() -> int:
    banner("SMOKE TEST  |  does the saved model actually work?")

    model_path = PROJECT_ROOT / "model.joblib"
    meta_path = PROJECT_ROOT / "model_metadata.json"

    # --- test 1: does it load? -------------------------------------------
    sub_banner("Test 1: load model.joblib")
    if not model_path.exists():
        print("  FAIL: model.joblib not found. Run 04_train_models.py first.")
        return 1

    model = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"  PASS  loaded {model_path.name} "
          f"({model_path.stat().st_size / 1024**2:.2f} MB)")
    print(f"        winning model : {metadata['model_name']}")
    print(f"        threshold     : {metadata['decision_threshold']}")
    print(f"        trained with  : scikit-learn "
          f"{metadata['library_versions']['scikit-learn']}")

    # --- test 2: does it accept one hand-typed order? --------------------
    sub_banner("Test 2: predict on two hand-written orders")
    expected = metadata["numeric_features"] + metadata["categorical_features"]

    results = {}
    for label, order in [("SAFE ", SAFE_ORDER), ("RISKY", RISKY_ORDER)]:
        missing = set(expected) - set(order)
        extra = set(order) - set(expected)
        if missing or extra:
            print(f"  FAIL: {label} order column mismatch.")
            if missing:
                print(f"        missing: {sorted(missing)}")
            if extra:
                print(f"        unexpected: {sorted(extra)}")
            return 1

        frame = pd.DataFrame([order])[expected]
        probability = float(model.predict_proba(frame)[0, 1])
        results[label.strip()] = probability

        verdict = "AT RISK" if probability >= metadata["decision_threshold"] else "LOW RISK"
        lift = probability / metadata["positive_class_rate"]
        print(f"  PASS  {label} order -> {probability * 100:5.1f}% chance of being late "
              f"  [{verdict:<8}]  = {lift:4.1f}x the average order")

    # --- test 3: is the ordering sensible? -------------------------------
    sub_banner("Test 3: does the model rank them the right way round?")
    if results["RISKY"] > results["SAFE"]:
        print(f"  PASS  risky order scores higher than safe order "
              f"({results['RISKY'] * 100:.1f}% > {results['SAFE'] * 100:.1f}%)")
    else:
        print("  FAIL  the model rates the safe order as riskier. Investigate.")
        return 1

    # --- test 4: unseen category must not crash --------------------------
    sub_banner("Test 4: an unseen product category must not crash the app")
    weird = dict(SAFE_ORDER)
    weird["primary_category"] = "quantum_flux_capacitors"   # does not exist
    weird["customer_state"] = "ZZ"                          # not a Brazilian state
    probability = float(model.predict_proba(pd.DataFrame([weird])[expected])[0, 1])
    print(f"  PASS  unknown category and state handled gracefully "
          f"-> {probability * 100:.1f}%")
    print("        (the OneHotEncoder bundles unseen values into a 'rare' bucket)")

    banner("ALL SMOKE TESTS PASSED")
    print("  The model is ready to be wired into the Streamlit app.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
