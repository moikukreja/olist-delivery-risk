/**
 * About.tsx
 * ---------
 * Project background, dataset provenance, team and links.
 */

import type { AppConfig } from "../types";
import { Glass, SectionHead } from "./ui";

const TEAM = [
  "Kartik Joshi",
  "Prem Kukreja",
  "Samuel Alex",
  "Gagandeep Singh",
  "Aditya Chitale",
  "Karan Baid",
];

const GITHUB_URL = "https://github.com/moikukreja/olist-delivery-risk";
const SPACE_URL = "https://huggingface.co/spaces/samuelalex37/olist-delivery-risk";
const KAGGLE_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce";

const PIPELINE_STEPS: [string, string][] = [
  ["Nine raw tables joined", "99,441 orders, 112,650 items, 3,095 sellers and 1,000,163 GPS readings merged into one modelling table of 96,470 delivered orders."],
  ["Distance engineered", "The single strongest geographic signal — kilometres between seller and buyer — was computed with the haversine formula. It does not exist in the raw data."],
  ["Leakage removed", "Seven columns that would only be known after delivery were deliberately dropped, including the delivery date the target is built from."],
  ["Three models compared", "Logistic Regression, Random Forest and Gradient Boosting — one per algorithm family. Gradient Boosting won on ROC-AUC."],
  ["Probabilities calibrated", "Isotonic regression corrected a 4.4× inflation caused by class weighting, so the percentages on screen mean what they say."],
  ["Threshold chosen by cost", "Not the default 0.5, and not max-F1 — the cut-off that minimises expected business cost, cutting it 28.9% versus doing nothing."],
];

export default function About({ config }: { config: AppConfig }) {
  const { reference, model } = config;

  return (
    <div className="fade-in">
      <div className="grid split-2" style={{ marginBottom: 16 }}>
        <Glass>
          <SectionHead title="The problem" sub="Why a marketplace should care about late parcels" />
          <div className="prose">
            <p>
              Olist connects thousands of small Brazilian sellers to customers across all
              27 states. When a seller delivers late,{" "}
              <b>the customer does not blame the seller — they blame Olist.</b>
            </p>
            <p>Our analysis of {reference.total_orders.toLocaleString()} delivered orders shows exactly how expensive that is:</p>
            <ul>
              <li>Orders delivered <b>on time</b> average <b>4.29 stars</b></li>
              <li>Orders delivered <b>late</b> average <b>2.57 stars</b></li>
              <li>
                The share of 1–2 star reviews jumps from <b>9.2%</b> to <b>54.1%</b> —
                a <b>5.9× increase</b>
              </li>
            </ul>

            <h3>The solution</h3>
            <p>
              This application scores every order <b>at the moment it is placed</b>, using
              only information available at checkout. Orders flagged as high risk can be
              given a realistic delivery estimate, upgraded shipping, or a proactive
              courtesy contact — <i>before</i> the customer is disappointed.
            </p>
            <p style={{ marginBottom: 0 }}>
              In effect, it predicts a customer-churn event at the one moment when it can
              still be prevented.
            </p>
          </div>
        </Glass>

        <Glass>
          <SectionHead title="How it was built" sub="Six steps from raw CSVs to a live prediction" />
          <div style={{ display: "grid", gap: 11 }}>
            {PIPELINE_STEPS.map(([title, detail], i) => (
              <div
                key={title}
                style={{
                  display: "flex",
                  gap: 13,
                  padding: "11px 13px",
                  borderRadius: 12,
                  background: "rgba(255,255,255,0.035)",
                }}
              >
                <div
                  style={{
                    minWidth: 26, height: 26, borderRadius: 8,
                    display: "grid", placeItems: "center",
                    fontSize: 12, fontWeight: 750, color: "#061024",
                    background: "linear-gradient(135deg, var(--accent), var(--accent-deep))",
                  }}
                >
                  {i + 1}
                </div>
                <div>
                  <div style={{ fontSize: 12.5, fontWeight: 650, marginBottom: 3 }}>{title}</div>
                  <div style={{ fontSize: 11.5, color: "var(--text-dim)", lineHeight: 1.6 }}>
                    {detail}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Glass>
      </div>

      <div className="grid split-3">
        <Glass>
          <SectionHead title="Dataset" />
          <div className="prose">
            <p>
              <b>Brazilian E-Commerce Public Dataset by Olist</b>
            </p>
            <p>
              Real, anonymised commercial data covering {reference.date_range[0]} to{" "}
              {reference.date_range[1]}. Olist anonymised it so thoroughly that company
              names in the customer reviews were replaced with Game of Thrones house names.
            </p>
            <ul>
              <li>99,441 orders across 9 linked tables</li>
              <li>Licence: CC BY-NC-SA 4.0</li>
              <li>Median order R$ {reference.median_order_value.toFixed(0)}</li>
              <li>Median journey {reference.median_distance_km.toFixed(0)} km</li>
            </ul>
            <p style={{ marginBottom: 0 }}>
              <a href={KAGGLE_URL} target="_blank" rel="noreferrer">
                View the dataset on Kaggle →
              </a>
            </p>
          </div>
        </Glass>

        <Glass>
          <SectionHead title="Technology" />
          <div className="prose">
            <h3 style={{ marginTop: 0 }}>Machine learning</h3>
            <ul>
              <li>scikit-learn {model.sklearnVersion} — pipeline, models, calibration</li>
              <li>pandas · NumPy · PyArrow</li>
              <li>Entire pipeline saved as one <code>model.joblib</code></li>
            </ul>
            <h3>Application</h3>
            <ul>
              <li>React 19 + TypeScript, built with Vite</li>
              <li>deck.gl — GPU-rendered WebGL choropleth</li>
              <li>Custom GLSL fragment shader for the background</li>
              <li>Recharts for the analytical charts</li>
              <li>FastAPI + Uvicorn serving both API and app</li>
            </ul>
            <h3>Deployment</h3>
            <ul style={{ marginBottom: 0 }}>
              <li>Docker container on Hugging Face Spaces</li>
              <li>GitHub Actions redeploys on every push to <code>main</code></li>
            </ul>
          </div>
        </Glass>

        <Glass>
          <SectionHead title="The team" />
          <div className="prose">
            <ul>
              {TEAM.map((member) => (
                <li key={member}>{member}</li>
              ))}
            </ul>
            <h3>Course</h3>
            <p>
              <b>Software Development for AI Models</b>
              <br />
              Instructor: Prof. JP Agarwal
              <br />
              Masters in AI with Business
            </p>
            <h3>Links</h3>
            <ul style={{ marginBottom: 0 }}>
              <li>
                <a href={GITHUB_URL} target="_blank" rel="noreferrer">
                  GitHub repository →
                </a>
              </li>
              <li>
                <a href={SPACE_URL} target="_blank" rel="noreferrer">
                  Hugging Face Space →
                </a>
              </li>
            </ul>
            <h3>A note on loading</h3>
            <p style={{ marginBottom: 0 }}>
              This Space runs on free hardware, which sleeps after 48 hours of
              inactivity. If you see <b>&ldquo;Starting&rdquo;</b>, give it about
              30 seconds — the container is waking up, not rebuilding.
            </p>
          </div>
        </Glass>
      </div>
    </div>
  );
}
