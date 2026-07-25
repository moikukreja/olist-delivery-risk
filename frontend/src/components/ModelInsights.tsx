/**
 * ModelInsights.tsx
 * -----------------
 * How the model was built, how well it performs, and what it relies on.
 *
 * This tab exists so a marker (or a sceptical colleague) can check our working
 * without opening a notebook. It deliberately includes the limitations as well
 * as the results.
 */

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AppConfig } from "../types";
import { Glass, Kpi, SectionHead } from "./ui";

const AXIS = { stroke: "#737d9e", fontSize: 11 };
const GRID = "rgba(255,255,255,0.06)";
const TOOLTIP_STYLE = {
  background: "rgba(6,10,28,0.95)",
  border: "1px solid rgba(255,255,255,0.2)",
  borderRadius: 12,
  fontSize: 12,
  color: "#eef2ff",
};

const METRIC_KEYS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"] as const;

const LEAKY_COLUMNS: [string, string][] = [
  ["order_delivered_customer_date", "The target is built from this date"],
  ["order_delivered_carrier_date", "Recorded days after checkout"],
  ["order_approved_at", "Recorded once payment clears"],
  ["order_status", "Only known after the order finishes"],
  ["review_score", "Written by the customer after delivery"],
  ["purchase_year", "Would not generalise to future orders"],
];

export default function ModelInsights({ config }: { config: AppConfig }) {
  const { model } = config;
  const rows = Object.entries(model.comparison).sort(
    (a, b) => b[1]["ROC-AUC"] - a[1]["ROC-AUC"]
  );
  const winner = rows[0][0];

  const importance = [...model.topFeatures].reverse();
  const maxImportance = Math.max(...model.topFeatures.map((f) => f.importance), 0.001);

  return (
    <div className="fade-in">
      <div className="grid kpi-row">
        <Kpi label="Winning model" value={model.name} note={`beat ${rows.length - 1} others`} />
        <Kpi
          label="ROC-AUC"
          value={model.metrics["ROC-AUC"].toFixed(3)}
          note="0.5 = guessing, 1.0 = perfect"
          accent="var(--accent)"
        />
        <Kpi
          label="Late orders caught"
          value={`${(model.metrics.Recall * 100).toFixed(1)}%`}
          note="recall on the held-out set"
        />
        <Kpi
          label="Action threshold"
          value={`${(model.threshold * 100).toFixed(1)}%`}
          note="chosen to minimise business cost"
        />
        <Kpi
          label="Training data"
          value={model.trainingRows.toLocaleString()}
          note={`tested on ${model.testRows.toLocaleString()} unseen orders`}
        />
      </div>

      {/* ---------------- model comparison ---------------- */}
      <Glass style={{ marginBottom: 16 }}>
        <SectionHead
          title="Three models, three algorithm families"
          sub="All scored on the same held-out 20% of orders that none of them saw during training."
        />
        <div style={{ overflowX: "auto" }}>
          <table className="table">
            <thead>
              <tr>
                <th>Model</th>
                {METRIC_KEYS.map((key) => (
                  <th key={key} style={{ textAlign: "right" }}>{key}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map(([name, metrics]) => (
                <tr key={name} className={name === winner ? "winner" : ""}>
                  <td>
                    {name}
                    {name === winner && (
                      <span style={{ color: "var(--accent)", marginLeft: 8, fontSize: 11 }}>
                        ★ selected
                      </span>
                    )}
                  </td>
                  {METRIC_KEYS.map((key) => (
                    <td key={key} className="num" style={{ textAlign: "right" }}>
                      {metrics[key].toFixed(3)}
                    </td>
                  ))}
                </tr>
              ))}
              <tr style={{ opacity: 0.6 }}>
                <td><i>“Always say on time”</i></td>
                <td className="num" style={{ textAlign: "right" }}>0.919</td>
                <td className="num" style={{ textAlign: "right" }}>—</td>
                <td className="num" style={{ textAlign: "right" }}>0.000</td>
                <td className="num" style={{ textAlign: "right" }}>—</td>
                <td className="num" style={{ textAlign: "right" }}>0.500</td>
                <td className="num" style={{ textAlign: "right" }}>0.081</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="section-sub" style={{ marginTop: 12, marginBottom: 0 }}>
          Note the bottom row: a model that simply answers “on time” every single time
          scores <b style={{ color: "var(--text)" }}>91.9% accuracy</b> while catching
          zero late deliveries. That is exactly why accuracy is not our headline metric —
          we rank on ROC-AUC and judge on recall.
        </p>
      </Glass>

      {/* ---------------- importance + leakage ---------------- */}
      <div className="grid split-2" style={{ marginBottom: 16 }}>
        <Glass>
          <SectionHead
            title="What the model actually relies on"
            sub="Permutation importance: how much ROC-AUC drops when a column is shuffled"
          />
          <ResponsiveContainer width="100%" height={380}>
            <BarChart
              layout="vertical"
              data={importance}
              margin={{ top: 4, right: 16, left: 66, bottom: 0 }}
            >
              <CartesianGrid stroke={GRID} horizontal={false} />
              <XAxis type="number" {...AXIS} tickLine={false} axisLine={false} />
              <YAxis
                type="category"
                dataKey="feature"
                {...AXIS}
                width={132}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
              <Bar dataKey="importance" name="ROC-AUC drop" radius={[0, 5, 5, 0]}>
                {importance.map((f, i) => (
                  <Cell
                    key={i}
                    fill={
                      f.importance / maxImportance > 0.6
                        ? "#6ee7ff"
                        : f.importance / maxImportance > 0.2
                        ? "#4f7cff"
                        : "#3a4a80"
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="section-sub" style={{ marginTop: 8, marginBottom: 0 }}>
            <b style={{ color: "var(--accent)" }}>distance_km</b> did not exist in the raw
            Kaggle data — we engineered it from 1,000,163 GPS readings using the haversine
            formula.
          </p>
        </Glass>

        <Glass>
          <SectionHead
            title="How data leakage was prevented"
            sub="The single most important design decision in the whole project"
          />
          <div className="prose">
            <p>
              <b>Data leakage</b> means letting the model see facts that would not
              exist yet at the moment we need the prediction. It is like letting a
              student see the exam answers: the score looks perfect and means nothing.
            </p>
            <p>
              Our rule: <b>the model may only use what the shop knows at checkout.</b>{" "}
              These columns were deliberately removed.
            </p>
          </div>
          <div style={{ overflowX: "auto", marginTop: 10 }}>
            <table className="table">
              <thead>
                <tr>
                  <th>Column removed</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {LEAKY_COLUMNS.map(([column, reason]) => (
                  <tr key={column}>
                    <td style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>{column}</td>
                    <td>{reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="section-sub" style={{ marginTop: 12, marginBottom: 0 }}>
            Every preprocessing step — filling gaps, scaling, encoding — is fitted on the
            training half only and stored inside the saved pipeline, so nothing about the
            test data can influence the model.
          </p>
        </Glass>
      </div>

      {/* ---------------- calibration + limitations ---------------- */}
      <div className="grid split-2">
        <Glass>
          <SectionHead
            title="Why the percentages can be trusted"
            sub="Probability calibration"
          />
          <div className="prose">
            <p>
              To make the model take rare late deliveries seriously we weighted them
              more heavily during training. That worked — but it also taught the model
              that late orders are far more common than they really are, so every
              percentage it printed was inflated by about <b>4.4×</b>.
            </p>
            <p>
              Orders the raw model scored “30–40% likely late” were in reality late only{" "}
              <b>3.4%</b> of the time. Shipping that into an app would have been
              dishonest.
            </p>
            <p>
              We fixed it with <b>isotonic calibration</b>, which learns a correction
              curve from the data. The average predicted probability fell from{" "}
              <b>36.0%</b> to <b>8.13%</b> against a true rate of <b>8.11%</b>, and the
              Brier score improved by 60% — while ROC-AUC was unchanged at 0.810.
            </p>
            <p style={{ marginBottom: 0 }}>
              That is why a reading of “20%” on this app really does mean roughly one
              order in five.
            </p>
          </div>
        </Glass>

        <Glass>
          <SectionHead title="Honest limitations" sub="What this model cannot do" />
          <div className="callout" style={{ marginBottom: 12 }}>
            <b style={{ color: "var(--text)" }}>
              Precision is {(model.metrics.Precision * 100).toFixed(0)}%.
            </b>{" "}
            Roughly {Math.round((1 - model.metrics.Precision) * 10)} in every 10 flagged
            orders would actually have arrived on time. This is acceptable only because
            the intervention is cheap — a courtesy email costs far less than the 1-star
            review that 54% of late deliveries attract. If intervention costs rise, the
            threshold should rise with them.
          </div>
          <div className="prose">
            <ul>
              <li>
                Trained on Brazilian marketplace data from 2016–2018. Courier networks
                change, so the model would need periodic retraining in real use.
              </li>
              <li>
                The February–March 2018 disruption is visible in the data but its cause
                is not recorded, so the model learns the pattern without understanding it.
              </li>
              <li>
                Seller-level history was excluded. It would likely help, but computing it
                safely requires time-aware aggregation to avoid reintroducing leakage.
              </li>
              <li>
                ROC-AUC of 0.810 means the ranking is good, not flawless. This is a
                decision-support tool, not an oracle.
              </li>
            </ul>
          </div>
          <p className="section-sub" style={{ marginTop: 6, marginBottom: 0 }}>
            Trained {new Date(model.trainedAt).toLocaleString()} · {model.calibration} ·
            scikit-learn {model.sklearnVersion}
          </p>
        </Glass>
      </div>
    </div>
  );
}
