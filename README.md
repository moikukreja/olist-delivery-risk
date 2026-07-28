---
title: Olist Delivery Risk Radar
emoji: 📦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-sa-4.0
short_description: Predicting late delivery risk for Brazilian e-commerce
---

# 📦 Olist Delivery Risk Radar

**Predicting late deliveries on Brazil's largest e-commerce marketplace.**

An end-to-end machine-learning application: nine raw Kaggle tables → a calibrated
gradient-boosting model → a React + WebGL dashboard, deployed automatically to
Hugging Face Spaces by GitHub Actions on every push to `main`.

> The block at the very top of this file is **YAML frontmatter**. Hugging Face
> reads it to learn how to run the Space — `sdk: docker` tells it to build our
> `Dockerfile`, and `app_port: 7860` tells it which port to expose. GitHub simply
> renders it as a small table, so one README serves both platforms.

**🔗 Live app:** https://huggingface.co/spaces/samuelalex37/olist-delivery-risk
**🔗 Source:** https://github.com/moikukreja/olist-delivery-risk

---

## The problem

Olist is a marketplace connecting thousands of small Brazilian sellers to
customers across all 27 states. When a seller delivers late, **the customer
blames Olist, not the seller.**

Our analysis of 96,470 delivered orders quantifies the damage:

| | On time | Late |
|---|---|---|
| Average review score | **4.29 ★** | **2.57 ★** |
| Share of 1–2 star reviews | **9.2%** | **54.1%** |

A late delivery makes a scathing review **5.9× more likely**. This app scores
every order **at the moment it is placed**, so at-risk orders can be given a
realistic estimate, upgraded shipping, or a proactive courtesy contact —
*before* the customer is disappointed.

---

## Results

| Model | Accuracy | Precision | Recall | F1 | **ROC-AUC** | PR-AUC |
|---|---|---|---|---|---|---|
| **Gradient Boosting** ★ | 0.777 | 0.219 | 0.686 | 0.332 | **0.810** | 0.336 |
| Random Forest | 0.845 | 0.273 | 0.551 | 0.365 | 0.799 | 0.330 |
| Logistic Regression | 0.657 | 0.144 | 0.655 | 0.236 | 0.710 | 0.183 |
| *"always on time"* | *0.919* | — | *0.000* | — | *0.500* | *0.081* |

Note the last row: a model that answers "on time" every single time scores
**91.9% accuracy** while catching **zero** late deliveries. That is exactly why
accuracy is not the headline metric — we rank on ROC-AUC and judge on recall.

### Three decisions worth reading about

1. **Data leakage was designed out.** Seven columns that only exist *after*
   delivery were removed, including the date the target is built from. The model
   may only use what the shop knows at checkout.

2. **The probabilities were wrong, so we fixed them.** Class weighting inflated
   every prediction by ~4.4×; orders scored "30–40% likely late" were really late
   3.4% of the time. Isotonic calibration brought the mean predicted probability
   from 36.0% to 8.13% against a true rate of 8.11%, cutting the Brier score 60%
   while leaving ROC-AUC unchanged.

3. **The threshold came from business cost, not from F1.** Maximising F1
   silently assumes a missed late delivery costs the same as a false alarm — which
   contradicts the review data above. Minimising `5 × misses + false alarms`
   instead gives a 0.155 cut-off that reduces expected cost **28.9%** versus
   flagging nothing.

**Honest limitation:** precision is 30%, so roughly 7 in 10 flagged orders would
have arrived fine. That is acceptable only because the intervention is cheap.

---

## The engineered feature

The raw data has no distance column. We built one by joining the geolocation
table — **1,000,163 GPS readings** — to both seller and customer postcodes and
applying the haversine formula.

| Journey | Late rate |
|---|---|
| Under 50 km | 6.5% |
| Over 1,500 km | **13.1%** |

It ranks in the model's top six features and did not exist before we made it.

---

## Two ways to predict

**One order at a time** — a form collecting the twelve facts known at checkout,
returning a calibrated probability, a risk tier and the reasoning behind it.

**A whole file at once** — upload a CSV and every row is scored in one pass.
2,000 orders take under half a second. Rows that fail validation report their own
line number while the rest still score. The results download with six added
columns: distance, probability, risk tier, flag, lift versus average, and the
recommended action.

Both paths share one feature-building function, so they cannot drift apart. The
downloadable template holds twelve example orders spanning all four payment
types, same-state and cross-country routes, and every risk band. Its first four
rows score 1.4% / 11.7% / 36.0% / 22.3% — identical to the four presets on the
single-order form, which keeps the two paths checkable against each other by
eye.

The batch endpoint is a plain HTTP API, so it works from a scheduled job too:

```bash
curl -F "file=@orders.csv" https://samuelalex37-olist-delivery-risk.hf.space/api/predict/batch
```

Every route is documented interactively at
[`/docs`](https://samuelalex37-olist-delivery-risk.hf.space/docs).

---

## Architecture

```
Browser  ──►  React 19 + TypeScript          (deck.gl WebGL map, GLSL shader background)
                     │  fetch /api/*
                     ▼
              FastAPI + Uvicorn              (app.py)
                     │
                     ▼
              model.joblib                   (full sklearn pipeline + isotonic calibration)
```

One container serves both the API and the compiled frontend on port 7860.

### Two workflows, two jobs

| Workflow | Trigger | Ships |
|---|---|---|
| `deploy.yml` | every push to `main` | the **application** |
| `mlops-pipeline.yml` | manual or scheduled | the **model** |

Retraining on every push would deploy a slightly different model each time and
the figures in this README would stop matching the live app. So the two are
separated deliberately.

The MLOps pipeline has four stages, each on its own machine, passing work
through the Hugging Face Hub:

```
register-dataset  →  data-prep  →  model-training  →  deploy-hosting
  HF Dataset          frozen         MLflow +           gated promotion
  repository          split          HF Model repo      to the live Space
```

Stage 4 only runs when explicitly requested, and first checks the model loads,
ranks a risky order above a safe one, and is not more than 0.02 ROC-AUC worse
than what is already live.

**Reproducibility:** the pipeline has been run three times — during original
development, locally through the scripts, and on a GitHub Actions runner from a
clean checkout. All three produced ROC-AUC 0.8101, threshold 0.155, and a 28.9%
cost reduction, on two different operating systems.

- 📊 [Registered dataset](https://huggingface.co/datasets/samuelalex37/olist-delivery-risk-data)
- 🤖 [Registered model](https://huggingface.co/samuelalex37/olist-delivery-risk-model)

### Repository layout

```
olist-delivery-risk/
├── app.py                     FastAPI backend + static file server
├── model.joblib               entire pipeline: impute → scale → encode → model → calibrate
├── model_metadata.json        metrics, threshold, feature list, library versions
├── app_config.json            state coordinates, route distances, dropdown options
├── dashboard_data.parquet     96,470 orders, 16 columns (2.9 MB)
├── requirements.txt           what the deployed app needs
├── requirements-dev.txt       plus plotting + the Streamlit fallback
├── Dockerfile                 two-stage: Node builds frontend → Python runs it
├── streamlit_app.py           earlier Streamlit prototype, kept as a fallback
├── frontend/                  React + TypeScript source
│   ├── src/components/        Dashboard, Predict, ModelInsights, About, BrazilMap, ShaderBackground
│   └── src/assets/            simplified Brazil GeoJSON (5.35 MB → 118 KB)
├── src/                       the ML pipeline, numbered in run order
│   ├── 01_data_audit.py       inspect the nine raw tables, change nothing
│   ├── 02_build_dataset.py    join 9 tables → 96,470 × 53, engineer distance
│   ├── 03_eda.py              11 report figures + key observations
│   ├── 04_train_models.py     preprocess, train 3 models, calibrate, save
│   ├── 05_smoke_test.py       prove model.joblib loads and predicts sanely
│   ├── 06_build_app_assets.py shrink 20.6 MB → 2.9 MB for deployment
│   └── 07_simplify_geojson.py Douglas-Peucker map simplification
├── mlops/                     the automated MLOps pipeline
│   ├── config.py              repo ids, feature contract, leakage guard
│   ├── data_register.py       stage 1 — publish the dataset to the Hub
│   ├── prep.py                stage 2 — freeze the train/test split
│   ├── train.py               stage 3 — train, calibrate, register
│   ├── hosting.py             stage 4 — pre-flight checks, then promote
│   └── requirements.txt       pipeline-only dependencies
├── reports/figures/           16 charts used in the report
├── report/                    the LaTeX report and compiled PDF
└── .github/workflows/
    ├── deploy.yml             ships the app on every push
    └── mlops-pipeline.yml     retrains the model on demand
```

---

## Running it locally

The raw Kaggle data is **not** committed — 120 MB does not belong in Git.
Download it from
[Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and unzip
all nine CSVs into `data/raw/`.

```bash
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Rebuild the model from scratch (about 10 minutes):

```bash
venv\Scripts\python.exe src\01_data_audit.py
```

```bash
venv\Scripts\python.exe src\02_build_dataset.py
```

```bash
venv\Scripts\python.exe src\03_eda.py
```

```bash
venv\Scripts\python.exe src\04_train_models.py
```

```bash
venv\Scripts\python.exe src\06_build_app_assets.py
```

Run the app (backend and frontend in two terminals):

```bash
venv\Scripts\python.exe -m uvicorn app:app --reload --port 7860
```

```bash
cd frontend && npm install && npm run dev
```

Then open http://localhost:5173. Or build the whole thing as one container:

```bash
docker build -t olist-risk . && docker run -p 7860:7860 olist-risk
```

---

## Dataset

**Brazilian E-Commerce Public Dataset by Olist** — real, anonymised commercial
data covering September 2016 to October 2018. Olist anonymised it so thoroughly
that company names in the review text were replaced with Game of Thrones house
names.

99,441 orders across 9 linked tables · Licence CC BY-NC-SA 4.0 ·
[View on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

---

## Team

Kartik Joshi · Prem Kukreja · Samuel Alex · Gagandeep Singh · Aditya Chitale · Karan Baid

**Software Development for AI Models** — Instructor: Prof. JP Agarwal
Masters in AI with Business
