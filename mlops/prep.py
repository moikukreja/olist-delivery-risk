"""
mlops/prep.py
-------------
STAGE 2 of the MLOps pipeline: split the registered data into train and test.

WHY THIS IS A SEPARATE STAGE
Splitting once, and publishing the result, means every future training run uses
exactly the same test set. If each training run made its own split, two models
could not be compared fairly - a difference in score might just mean one got an
easier exam. Freezing the split turns model comparison into a controlled
experiment.

The split is STRATIFIED, so the 8.11% late rate is preserved in both halves.

Run it with:
    venv\\Scripts\\python.exe mlops\\prep.py
"""

import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def main() -> int:
    config.banner("STAGE 2  |  DATA PREPARATION AND TRAIN/TEST SPLIT")

    api = HfApi(token=config.token())

    # ---- read the registered dataset straight from the Hub ----------------
    # The hf:// protocol lets pandas read from the Hub as if it were a local
    # path, so nothing has to be downloaded and managed by hand.
    source = f"hf://datasets/{config.DATASET_REPO}/{config.ANALYTICAL_FILE}"
    print(f"  Reading {source}")
    frame = pd.read_parquet(source)
    print(f"  Loaded {len(frame):,} orders x {frame.shape[1]} columns")

    # ---- guard against leakage before anything else -----------------------
    # This is a hard failure, not a warning. If a banned column ever reappears
    # in the analytical table, the pipeline must stop rather than quietly train
    # a model that cheats.
    present = [c for c in config.BANNED_COLUMNS if c in config.FEATURES]
    if present:
        print(f"\n  ERROR: banned column(s) found in the feature list: {present}")
        return 1
    print(f"  Leakage guard passed: none of the {len(config.BANNED_COLUMNS)} "
          f"banned columns appear in the feature list")

    missing = [c for c in config.FEATURES + [config.TARGET] if c not in frame.columns]
    if missing:
        print(f"\n  ERROR: the dataset is missing expected column(s): {missing}")
        return 1

    X = frame[config.FEATURES]
    y = frame[config.TARGET]

    print(f"\n  Features : {len(config.NUMERIC_FEATURES)} numeric + "
          f"{len(config.CATEGORICAL_FEATURES)} categorical = {len(config.FEATURES)}")
    print(f"  Target   : {config.TARGET}  ({y.mean() * 100:.2f}% positive)")

    # ---- stratified split --------------------------------------------------
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    print(f"\n  Training : {len(Xtrain):>7,} orders   late rate {ytrain.mean() * 100:.2f}%")
    print(f"  Test     : {len(Xtest):>7,} orders   late rate {ytest.mean() * 100:.2f}%")
    print("  Stratification kept the class balance identical in both halves.")

    # ---- publish the splits back to the Hub -------------------------------
    # Parquet for the feature matrices because it preserves dtypes exactly;
    # CSV for the single-column targets because they are tiny and readable.
    work = Path("mlops_artifacts")
    work.mkdir(exist_ok=True)

    outputs = [
        (config.XTRAIN_FILE, Xtrain, "parquet"),
        (config.XTEST_FILE, Xtest, "parquet"),
        (config.YTRAIN_FILE, ytrain, "csv"),
        (config.YTEST_FILE, ytest, "csv"),
    ]

    print()
    for name, data, kind in outputs:
        path = work / name
        if kind == "parquet":
            data.to_parquet(path, index=False)
        else:
            data.to_csv(path, index=False)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=config.DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Publish {name} from stage 2 (data preparation)",
        )
        print(f"  Uploaded {name:<20} {path.stat().st_size / 1024:>8,.0f} KB")

    print(f"\n  https://huggingface.co/datasets/{config.DATASET_REPO}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(config.run_stage(main, config.DATASET_REPO, "dataset"))
