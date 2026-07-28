"""
mlops/data_register.py
----------------------
STAGE 1 of the MLOps pipeline: register the dataset on the Hugging Face Hub.

WHY THIS STAGE EXISTS
The raw Kaggle files are 120 MB and deliberately kept out of Git. But the four
pipeline stages each run on a different, disposable machine, so they cannot
pass files to one another directly. The Hub becomes the shared storage layer:
this stage publishes the analytical table once, and every later stage reads it
back from there.

That also makes the data VERSIONED. Every upload creates a commit on the
dataset repository, so a model can always be traced back to the exact data it
was trained on - which is the whole point of a registry.

HOW IT BEHAVES IN THE TWO PLACES IT RUNS
  - On a laptop, where data/processed/olist_analytical.parquet exists, it
    uploads that file (this is how the dataset is first seeded).
  - On a GitHub runner, where the file does not exist, it verifies the dataset
    is already registered and moves on rather than failing the pipeline.

Run it with:
    venv\\Scripts\\python.exe mlops\\data_register.py
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cards  # noqa: E402
import config  # noqa: E402

LOCAL_PARQUET = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / config.ANALYTICAL_FILE
)


def main() -> int:
    config.banner("STAGE 1  |  REGISTER DATASET ON THE HUGGING FACE HUB")

    api = HfApi(token=config.token())

    # ---- ensure the dataset repository exists -----------------------------
    try:
        api.repo_info(repo_id=config.DATASET_REPO, repo_type="dataset")
        print(f"  Dataset repo '{config.DATASET_REPO}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"  Dataset repo '{config.DATASET_REPO}' not found. Creating it...")
        try:
            create_repo(
                repo_id=config.DATASET_REPO,
                repo_type="dataset",
                private=False,
                token=config.token(),
            )
            print("  Created.")
        except Exception as exc:
            code = config.explain_permission_error(exc, config.DATASET_REPO, "dataset")
            if code >= 0:
                return code
            raise

    # ---- upload the analytical table if we have it locally ----------------
    if LOCAL_PARQUET.exists():
        size_mb = LOCAL_PARQUET.stat().st_size / 1024**2
        print(f"\n  Found local analytical table ({size_mb:.1f} MB). Uploading...")
        try:
            api.upload_file(
                path_or_fileobj=str(LOCAL_PARQUET),
                path_in_repo=config.ANALYTICAL_FILE,
                repo_id=config.DATASET_REPO,
                repo_type="dataset",
                commit_message="Register analytical table (96,470 delivered orders)",
            )
            print("  Uploaded.")
        except Exception as exc:
            code = config.explain_permission_error(exc, config.DATASET_REPO, "dataset")
            if code >= 0:
                return code
            raise
    else:
        # This is the normal path inside GitHub Actions, where the 120 MB of raw
        # Kaggle data is not present. The dataset must already be registered.
        print(f"\n  No local copy at {LOCAL_PARQUET.name} - this is expected in CI.")
        print("  Verifying the dataset is already registered on the Hub...")
        files = api.list_repo_files(repo_id=config.DATASET_REPO, repo_type="dataset")
        if config.ANALYTICAL_FILE not in files:
            print(f"\n  ERROR: '{config.ANALYTICAL_FILE}' is not in the dataset repo,")
            print("  and no local copy exists to upload.")
            print("  Seed it once from a machine that has the data:")
            print("      venv\\Scripts\\python.exe mlops\\data_register.py")
            return 1
        print("  Present. Nothing to upload.")

    # ---- publish the dataset card -----------------------------------------
    # The card is not decoration. Its frontmatter declares three separate
    # configurations, which stops the Hub viewer concatenating the full table
    # with the train/test partitions derived from it and reporting 192,940 rows
    # for a dataset that contains 96,470 orders.
    print("\n  Publishing the dataset card...")
    try:
        api.upload_file(
            path_or_fileobj=cards.DATASET_CARD.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=config.DATASET_REPO,
            repo_type="dataset",
            commit_message="Add dataset card declaring separate configurations",
        )
        print("  Published. The viewer will now show each configuration separately.")
    except Exception as exc:
        code = config.explain_permission_error(exc, config.DATASET_REPO, "dataset")
        if code >= 0:
            return code
        raise

    # ---- report what the repository now holds -----------------------------
    files = api.list_repo_files(repo_id=config.DATASET_REPO, repo_type="dataset")
    print(f"\n  Dataset repository now contains {len(files)} file(s):")
    for name in sorted(files):
        if not name.startswith("."):
            print(f"    - {name}")

    print(f"\n  https://huggingface.co/datasets/{config.DATASET_REPO}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
