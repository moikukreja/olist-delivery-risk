"""
mlops/train.py
--------------
STAGE 3 of the MLOps pipeline: train, calibrate, evaluate and register a model.

This stage reproduces the modelling decisions documented in the report, but as
an unattended script rather than an interactive session:

  1. Read the frozen train/test split from the Hub
  2. Train three models, one per algorithm family
  3. Log every run to MLflow
  4. Calibrate the winner so its probabilities are honest
  5. Choose the decision threshold by expected business cost
  6. Publish the model and its metadata to the Hub model registry

WHY MLFLOW IS USED THE WAY IT IS
The course material starts an MLflow server inside the runner and logs to
localhost. That works, but the runner is destroyed when the job ends, taking
the tracking database with it. Here MLflow writes to a local ./mlruns
directory instead, which the workflow then uploads as a build artefact. The
experiment history therefore survives the job and can be downloaded and opened
afterwards. Honest limitation: this is still per-run history, not a permanent
central tracking server, which a production team would host separately.

Run it with:
    venv\\Scripts\\python.exe mlops\\train.py
"""

import json
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cards  # noqa: E402
import config  # noqa: E402


def build_preprocessor() -> ColumnTransformer:
    """Numbers and words need different treatment, so they take parallel paths."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=30,
            sparse_output=False,
        )),
    ])
    return ColumnTransformer(
        [("num", numeric, config.NUMERIC_FEATURES),
         ("cat", categorical, config.CATEGORICAL_FEATURES)],
        remainder="drop",
    )


def build_models() -> dict:
    """One model per algorithm family, so the comparison is meaningful."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, max_features="sqrt",
            class_weight="balanced", n_jobs=-1, random_state=config.RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
            early_stopping=True, validation_fraction=0.1,
            class_weight="balanced", random_state=config.RANDOM_STATE),
    }


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def choose_threshold(y_true, proba) -> tuple[float, float, float]:
    """
    Pick the cut-off that minimises expected business cost.

    Not the default 0.5, and not maximum F1 - both of those implicitly assume a
    missed late delivery costs the same as a false alarm, which the exploratory
    analysis showed is untrue.
    """
    grid = np.linspace(0.02, 0.95, 187)
    best_threshold, best_cost = 0.5, float("inf")
    for threshold in grid:
        tn, fp, fn, tp = confusion_matrix(
            y_true, (proba >= threshold).astype(int), labels=[0, 1]).ravel()
        cost = config.COST_RATIO * fn + fp
        if cost < best_cost:
            best_threshold, best_cost = float(threshold), float(cost)
    do_nothing = config.COST_RATIO * int(np.sum(y_true))
    return best_threshold, best_cost, do_nothing


def main() -> int:
    config.banner("STAGE 3  |  MODEL TRAINING, CALIBRATION AND REGISTRATION")

    api = HfApi(token=config.token())

    # MLflow 3.14 put the old ./mlruns filesystem store into maintenance mode
    # and now refuses to use it. SQLite is the supported replacement: still a
    # single self-contained file, no server to run, but on the actively
    # maintained code path. Both the database and the artefacts are uploaded by
    # the workflow so the experiment history outlives the disposable runner.
    tracking_db = Path("mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    mlflow.set_experiment("olist-delivery-risk")
    print(f"  MLflow tracking to {tracking_db.name} (uploaded as a build artefact)")
    print(f"  View it later with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")

    # ---- read the frozen split from the Hub -------------------------------
    base = f"hf://datasets/{config.DATASET_REPO}"
    print(f"\n  Reading the frozen split from {config.DATASET_REPO}")
    Xtrain = pd.read_parquet(f"{base}/{config.XTRAIN_FILE}")
    Xtest = pd.read_parquet(f"{base}/{config.XTEST_FILE}")
    ytrain = pd.read_csv(f"{base}/{config.YTRAIN_FILE}").squeeze("columns")
    ytest = pd.read_csv(f"{base}/{config.YTEST_FILE}").squeeze("columns")
    print(f"  Train {len(Xtrain):,}   Test {len(Xtest):,}   "
          f"late rate {ytrain.mean() * 100:.2f}% / {ytest.mean() * 100:.2f}%")

    # ---- train and compare ------------------------------------------------
    preprocessor = build_preprocessor()
    results, fitted = {}, {}

    config.banner("Training three models")
    for name, estimator in build_models().items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline([("prep", preprocessor), ("model", estimator)])
            pipeline.fit(Xtrain, ytrain)

            proba = pipeline.predict_proba(Xtest)[:, 1]
            metrics = evaluate(ytest, pipeline.predict(Xtest), proba)

            mlflow.log_param("model_family", name)
            mlflow.log_param("n_features", len(config.FEATURES))
            mlflow.log_param("n_training_rows", len(Xtrain))
            mlflow.log_metrics(metrics)

            results[name] = metrics
            fitted[name] = pipeline
            print(f"  {name:<22} ROC-AUC {metrics['roc_auc']:.4f}   "
                  f"recall {metrics['recall']:.4f}")

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\n  >>> Best by ROC-AUC: {best_name} <<<")

    # ---- calibrate ---------------------------------------------------------
    config.banner("Calibrating the winner")
    raw_proba = fitted[best_name].predict_proba(Xtest)[:, 1]
    print(f"  Mean raw predicted probability : {raw_proba.mean() * 100:5.2f}%")
    print(f"  True late rate                 : {ytest.mean() * 100:5.2f}%")
    print(f"  -> overstated by roughly {raw_proba.mean() / ytest.mean():.1f}x\n")

    with mlflow.start_run(run_name=f"{best_name} (calibrated)"):
        calibrated = CalibratedClassifierCV(fitted[best_name], method="isotonic", cv=5)
        calibrated.fit(Xtrain, ytrain)
        proba = calibrated.predict_proba(Xtest)[:, 1]

        threshold, cost, do_nothing = choose_threshold(ytest, proba)
        predictions = (proba >= threshold).astype(int)
        metrics = evaluate(ytest, predictions, proba)
        metrics["brier"] = brier_score_loss(ytest, proba)

        print(f"  Mean calibrated probability    : {proba.mean() * 100:5.2f}%")
        print(f"  Brier {brier_score_loss(ytest, raw_proba):.4f} -> {metrics['brier']:.4f}")
        print(f"  ROC-AUC unchanged at {metrics['roc_auc']:.4f}")
        print(f"\n  Cost-optimal threshold : {threshold:.3f}")
        print(f"  Expected cost          : {cost:,.0f} vs {do_nothing:,.0f} doing nothing "
              f"({(1 - cost / do_nothing) * 100:.1f}% lower)")

        mlflow.log_param("calibration", "isotonic (cv=5)")
        mlflow.log_param("decision_threshold", round(threshold, 4))
        mlflow.log_param("cost_ratio", config.COST_RATIO)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("cost_reduction_pct", (1 - cost / do_nothing) * 100)

        # ---- save artefacts ------------------------------------------------
        work = Path("mlops_artifacts")
        work.mkdir(exist_ok=True)
        model_path = work / config.MODEL_FILE
        joblib.dump(calibrated, model_path, compress=3)

        metadata = {
            "model_name": best_name,
            "calibration": "isotonic (CalibratedClassifierCV, cv=5)",
            "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "n_training_rows": int(len(Xtrain)),
            "n_test_rows": int(len(Xtest)),
            "target": config.TARGET,
            "positive_class_rate": float(pd.concat([ytrain, ytest]).mean()),
            "decision_threshold": round(threshold, 4),
            "cost_ratio": config.COST_RATIO,
            "numeric_features": config.NUMERIC_FEATURES,
            "categorical_features": config.CATEGORICAL_FEATURES,
            "test_metrics_shipped": {
                "Accuracy": round(metrics["accuracy"], 4),
                "Precision": round(metrics["precision"], 4),
                "Recall": round(metrics["recall"], 4),
                "F1": round(metrics["f1"], 4),
                "ROC-AUC": round(metrics["roc_auc"], 4),
                "PR-AUC": round(metrics["pr_auc"], 4),
                "Brier": round(metrics["brier"], 4),
            },
            "all_models": {
                name: {
                    "Accuracy": round(m["accuracy"], 4),
                    "Precision": round(m["precision"], 4),
                    "Recall": round(m["recall"], 4),
                    "F1": round(m["f1"], 4),
                    "ROC-AUC": round(m["roc_auc"], 4),
                    "PR-AUC": round(m["pr_auc"], 4),
                }
                for name, m in results.items()
            },
            "library_versions": {
                "scikit-learn": __import__("sklearn").__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
            "registered_from": "mlops/train.py",
            "dataset_repo": config.DATASET_REPO,
            "model_repo": config.MODEL_REPO,
        }
        metadata_path = work / config.METADATA_FILE
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metadata_path))

    # ---- register on the Hub ----------------------------------------------
    config.banner("Registering the model on the Hugging Face Hub")
    try:
        api.repo_info(repo_id=config.MODEL_REPO, repo_type="model")
        print(f"  Model repo '{config.MODEL_REPO}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"  Model repo '{config.MODEL_REPO}' not found. Creating it...")
        create_repo(repo_id=config.MODEL_REPO, repo_type="model",
                    private=False, token=config.token())
        print("  Created.")

    message = (f"{best_name} calibrated, ROC-AUC "
               f"{metadata['test_metrics_shipped']['ROC-AUC']}, "
               f"threshold {threshold:.3f}")
    for path, name in ((model_path, config.MODEL_FILE),
                       (metadata_path, config.METADATA_FILE)):
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=config.MODEL_REPO,
            repo_type="model",
            commit_message=message,
        )
        print(f"  Uploaded {name:<24} {path.stat().st_size / 1024:>8,.0f} KB")

    # The card is generated from THIS run's metrics, so it can never describe a
    # model other than the one sitting beside it.
    api.upload_file(
        path_or_fileobj=cards.model_card(metadata).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=config.MODEL_REPO,
        repo_type="model",
        commit_message=f"Model card for {message}",
    )
    print(f"  Uploaded {'README.md (model card)':<24}")

    config.banner("STAGE 3 COMPLETE")
    print(f"  Model    : {best_name}")
    print(f"  ROC-AUC  : {metadata['test_metrics_shipped']['ROC-AUC']}")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  Registry : https://huggingface.co/{config.MODEL_REPO}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(config.run_stage(main, config.MODEL_REPO, "model"))
