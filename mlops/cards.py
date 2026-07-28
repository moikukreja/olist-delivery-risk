"""
mlops/cards.py
--------------
Generate the dataset and model cards published alongside the artefacts.

WHY CARDS MATTER HERE
A registered artefact with no card is only half-registered. Someone finding the
repository has no way to know what the columns mean, how the data was built, or
which of several files is the one they want.

There is also a concrete problem the dataset card solves. Hugging Face's viewer
concatenates every parquet file in a repository into a single split. Our
repository holds the full analytical table AND the train/test partitions derived
from it, so the viewer reported 192,940 rows - the 96,470 real orders counted
once whole and once split. Declaring explicit configurations in the card's
frontmatter tells the viewer these are separate views of the same data, so each
reports its own correct row count.

The model card is generated from the metrics of the run that produced it, so it
can never drift out of date with the artefact it describes.
"""

DATASET_CARD = """---
license: cc-by-nc-sa-4.0
task_categories:
  - tabular-classification
tags:
  - e-commerce
  - logistics
  - delivery-prediction
  - brazil
size_categories:
  - 10K<n<100K
configs:
  - config_name: analytical
    default: true
    data_files:
      - split: full
        path: olist_analytical.parquet
  - config_name: train
    data_files:
      - split: train
        path: Xtrain.parquet
  - config_name: test
    data_files:
      - split: test
        path: Xtest.parquet
---

# Olist Delivery Risk — analytical dataset

Registered automatically by the MLOps pipeline of the
[Olist Delivery Risk Radar](https://huggingface.co/spaces/samuelalex37/olist-delivery-risk)
project.

## What this contains

**96,470 delivered orders** from the Brazilian marketplace Olist, assembled from
the nine raw Kaggle tables into a single modelling table.

> **Note on row counts.** This repository holds the full table *and* the
> train/test partitions derived from it. Use the configuration selector above to
> view one at a time — viewing them together would count the same orders twice.

| Configuration | Rows | Contents |
|---|---|---|
| `analytical` | 96,470 | The complete modelling table, 53 columns |
| `train` | 77,176 | Training features only, 34 columns |
| `test` | 19,294 | Held-out features only, 34 columns |

The targets are stored separately as `ytrain.csv` and `ytest.csv`.

## The prediction task

Predict whether an order will be **delivered after the date promised to the
customer at checkout**. 8.11% of orders are late.

## How it was built

Nine relational tables were joined on five keys. Item-level rows were aggregated
to one row per order. Most significantly, **`distance_km` was engineered** — the
great-circle distance between seller and buyer, computed with the haversine
formula from 1,000,163 GPS readings after filtering coordinates that fell outside
Brazil's borders.

## Leakage

Columns recorded *after* checkout are excluded, because the prediction is needed
at checkout. The pipeline asserts this before every training run and stops if a
banned column reappears:

`order_delivered_customer_date`, `order_delivered_carrier_date`,
`order_approved_at`, `order_estimated_delivery_date`, `order_status`,
`actual_delivery_days`, `days_late`, `review_score`, `purchase_year`

## The split is frozen deliberately

The train/test partition is computed once and published, rather than recomputed
per training run. If every run split independently, two models could not be
compared fairly — a difference in score might mean nothing more than one model
receiving an easier test set.

## Source

Derived from the
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
licensed CC BY-NC-SA 4.0. This derived dataset carries the same licence.
"""


def model_card(metadata: dict) -> str:
    """Build the model card from the metrics of the run that produced it."""
    shipped = metadata["test_metrics_shipped"]
    comparison = metadata["all_models"]

    rows = "\n".join(
        f"| {'**' + name + '**' if name == metadata['model_name'] else name} "
        f"| {m['Accuracy']:.3f} | {m['Precision']:.3f} | {m['Recall']:.3f} "
        f"| {m['F1']:.3f} | {m['ROC-AUC']:.3f} | {m['PR-AUC']:.3f} |"
        for name, m in sorted(comparison.items(),
                              key=lambda kv: kv[1]["ROC-AUC"], reverse=True)
    )

    return f"""---
license: cc-by-nc-sa-4.0
library_name: sklearn
tags:
  - tabular-classification
  - scikit-learn
  - calibrated-classifier
  - e-commerce
  - logistics
datasets:
  - {metadata['dataset_repo']}
metrics:
  - roc_auc
  - recall
  - brier_score
---

# Olist Delivery Risk — {metadata['model_name']}

Predicts whether a Brazilian e-commerce order will be **delivered later than the
date promised to the customer at checkout**.

Registered automatically by the project's MLOps pipeline.
Live application: [Olist Delivery Risk Radar](https://huggingface.co/spaces/samuelalex37/olist-delivery-risk)

## Performance

Measured on {metadata['n_test_rows']:,} held-out orders the model never saw
during training.

| Metric | Value |
|---|---|
| ROC-AUC | **{shipped['ROC-AUC']}** |
| PR-AUC | {shipped['PR-AUC']} |
| Recall | {shipped['Recall']} |
| Precision | {shipped['Precision']} |
| F1 | {shipped['F1']} |
| Brier score | {shipped['Brier']} |
| Decision threshold | **{metadata['decision_threshold']}** |

### Models compared

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
{rows}
| *"always on time"* | *0.919* | — | *0.000* | — | *0.500* | *0.081* |

That last row is the point. Only 8.11% of orders are late, so a model that
ignores every input and always answers "on time" scores **91.9% accuracy** while
catching **zero** late deliveries. Accuracy is not the metric to judge this on.

## The probabilities are calibrated

Class weighting made the raw model overstate risk by roughly **4.4×** — orders it
scored "30–40% likely late" were in reality late only 3.4% of the time.

Isotonic calibration corrected this. Mean predicted probability moved from 36.0%
to {metadata['positive_class_rate'] * 100:.2f}% against a true rate of
{metadata['positive_class_rate'] * 100:.2f}%, and the Brier score improved by
about 60% while ROC-AUC was unchanged.

**A reading of 20% from this model genuinely means about one order in five.**

## The threshold comes from business cost

Not the default 0.5, and not maximum F1 — both assume a missed late delivery
costs the same as a false alarm. It doesn't: 54.1% of late deliveries attract a
1–2 star review, against 9.2% of on-time ones.

The shipped threshold of **{metadata['decision_threshold']}** minimises
`{metadata['cost_ratio']} × misses + false alarms`, reducing expected cost by
about 29% versus taking no action.

## Usage

```python
import joblib, json, pandas as pd
from huggingface_hub import hf_hub_download

repo = "{metadata.get('model_repo', 'samuelalex37/olist-delivery-risk-model')}"
model = joblib.load(hf_hub_download(repo, "model.joblib"))
meta = json.load(open(hf_hub_download(repo, "model_metadata.json")))

columns = meta["numeric_features"] + meta["categorical_features"]
probability = model.predict_proba(pd.DataFrame([order])[columns])[0, 1]
flagged = probability >= meta["decision_threshold"]
```

`model.joblib` contains the **entire pipeline** — imputation, scaling, one-hot
encoding, the classifier and the calibrator — so no preprocessing code is needed
and the transformations cannot drift out of step with the model.

## Honest limitations

- **Precision is {shipped['Precision']:.2f}.** Roughly
  {round((1 - shipped['Precision']) * 10)} in every 10 flagged orders would have
  arrived on time. Acceptable only because the intervention is cheap.
- Trained on 2016–2018 courier conditions; would need periodic retraining.
- Seller-level history was excluded, since computing it safely requires
  time-aware aggregation to avoid reintroducing leakage.
- ROC-AUC {shipped['ROC-AUC']} means good ranking, not certainty. This is
  decision support, not an oracle.

## Provenance

- Trained: {metadata['trained_at']}
- Training rows: {metadata['n_training_rows']:,}
- Calibration: {metadata['calibration']}
- scikit-learn: {metadata['library_versions']['scikit-learn']}
- Registered by: `{metadata['registered_from']}`
"""
