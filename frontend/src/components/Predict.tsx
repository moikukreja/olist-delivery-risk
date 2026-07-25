/**
 * Predict.tsx
 * -----------
 * Score a single order at the moment of checkout.
 *
 * The user fills in 12 fields. The backend turns those into the 34 columns the
 * model was trained on, runs the saved pipeline, and returns a calibrated
 * probability - meaning "20%" really does mean roughly 1 order in 5.
 */

import { useMemo, useState } from "react";
import { predictOrder } from "../api";
import type { AppConfig, OrderRequest, PredictionResult } from "../types";
import { ErrorBox, Field, Glass, SectionHead, prettify, tierColour } from "./ui";

/**
 * Four ready-made orders, one for each risk band, so the whole scoring range
 * can be demonstrated in four clicks without typing anything.
 */
const PRESETS: {
  key: string;
  title: string;
  detail: string;
  order: Omit<OrderRequest, "purchaseDate"> & { purchaseDate: string };
}[] = [
  {
    key: "typical",
    title: "🟢 Typical order",
    detail: "São Paulo to São Paulo, 20-day promise, quiet month",
    order: {
      sellerState: "SP", customerState: "SP", promisedDays: 20,
      category: "housewares", orderValue: 90, freightValue: 17, weightGrams: 750,
      itemCount: 1, sellerCount: 1, paymentType: "credit_card", installments: 2,
      purchaseDate: "2018-06-15",
    },
  },
  {
    key: "moderate",
    title: "🟡 Long haul, fair promise",
    detail: "São Paulo to Pernambuco with a reasonable 12-day window",
    order: {
      sellerState: "SP", customerState: "PE", promisedDays: 12,
      category: "health_beauty", orderValue: 250, freightValue: 28, weightGrams: 1200,
      itemCount: 1, sellerCount: 1, paymentType: "credit_card", installments: 3,
      purchaseDate: "2018-08-20",
    },
  },
  {
    key: "aggressive",
    title: "🟠 Aggressive promise",
    detail: "Local delivery, but pledged in only 6 days",
    order: {
      sellerState: "SP", customerState: "SP", promisedDays: 6,
      category: "health_beauty", orderValue: 250, freightValue: 28, weightGrams: 1200,
      itemCount: 1, sellerCount: 1, paymentType: "credit_card", installments: 3,
      purchaseDate: "2018-08-20",
    },
  },
  {
    key: "blackfriday",
    title: "🔴 Black Friday long haul",
    detail: "São Paulo to Amazonas in the busiest week of the year",
    order: {
      sellerState: "SP", customerState: "AM", promisedDays: 12,
      category: "furniture_decor", orderValue: 480, freightValue: 42, weightGrams: 2000,
      itemCount: 1, sellerCount: 1, paymentType: "credit_card", installments: 8,
      purchaseDate: "2018-11-24",
    },
  },
];

/** A semicircular dial showing where the probability sits on the risk scale. */
function Gauge({ value, threshold, colour }: { value: number; threshold: number; colour: string }) {
  const MAX = 60;               // beyond 60% is vanishingly rare, so we cap here
  const clamped = Math.min(value, MAX);
  const radius = 88;
  const circumference = Math.PI * radius;      // half a circle
  const filled = (clamped / MAX) * circumference;
  const markerAngle = 180 - (threshold / MAX) * 180;
  const markerRad = (markerAngle * Math.PI) / 180;

  return (
    <svg viewBox="0 0 220 128" style={{ width: "100%", maxWidth: 300 }}>
      {/* the empty track */}
      <path
        d="M 22 116 A 88 88 0 0 1 198 116"
        fill="none"
        stroke="rgba(255,255,255,0.09)"
        strokeWidth="15"
        strokeLinecap="round"
      />
      {/* the filled portion */}
      <path
        d="M 22 116 A 88 88 0 0 1 198 116"
        fill="none"
        stroke={colour}
        strokeWidth="15"
        strokeLinecap="round"
        strokeDasharray={`${filled} ${circumference}`}
        style={{ transition: "stroke-dasharray .75s cubic-bezier(.22,1,.36,1)" }}
      />
      {/* where the action threshold sits */}
      <line
        x1={110 + Math.cos(markerRad) * 74}
        y1={116 - Math.sin(markerRad) * 74}
        x2={110 + Math.cos(markerRad) * 102}
        y2={116 - Math.sin(markerRad) * 102}
        stroke="#eef2ff"
        strokeWidth="2.5"
        strokeLinecap="round"
      />
      <text x="110" y="96" textAnchor="middle" fontSize="34" fontWeight="800" fill="#eef2ff">
        {value.toFixed(1)}%
      </text>
      <text x="110" y="116" textAnchor="middle" fontSize="10" fill="#737d9e">
        action threshold {threshold.toFixed(1)}%
      </text>
    </svg>
  );
}

interface Props {
  config: AppConfig;
}

export default function Predict({ config }: Props) {
  const [activePreset, setActivePreset] = useState("typical");
  const [order, setOrder] = useState<OrderRequest>(PRESETS[0].order);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function update<K extends keyof OrderRequest>(key: K, value: OrderRequest[K]) {
    setOrder((o) => ({ ...o, [key]: value }));
    setActivePreset("");           // the form no longer matches any preset
  }

  function applyPreset(key: string) {
    const preset = PRESETS.find((p) => p.key === key);
    if (!preset) return;
    setOrder(preset.order);
    setActivePreset(key);
    setResult(null);
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setResult(await predictOrder(order));
    } catch (e) {
      setError((e as Error).message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  // Show the user how far the parcel will travel before they even submit.
  const previewDistance = useMemo(() => {
    const key = `${order.sellerState}|${order.customerState}`;
    return config.routeDistances[key];
  }, [order.sellerState, order.customerState, config.routeDistances]);

  const colour = result ? tierColour(result.tier) : "var(--accent)";

  return (
    <div className="fade-in">
      <Glass style={{ marginBottom: 16 }}>
        <SectionHead
          title="Score an order at checkout"
          sub="Fill in only what the shop knows the moment the customer clicks Buy. Nothing recorded after that point is used."
        />

        <div className="preset-row">
          {PRESETS.map((preset) => (
            <button
              key={preset.key}
              type="button"
              className={`preset ${activePreset === preset.key ? "on" : ""}`}
              onClick={() => applyPreset(preset.key)}
            >
              <div className="t">{preset.title}</div>
              <div className="d">{preset.detail}</div>
            </button>
          ))}
        </div>

        <form onSubmit={submit}>
          <div className="grid split-3">
            {/* ---- route ---- */}
            <div>
              <div className="kpi-label" style={{ marginBottom: 12 }}>🗺️ Route</div>
              <div className="grid" style={{ gap: 12 }}>
                <Field label="Seller is in">
                  <select
                    value={order.sellerState}
                    onChange={(e) => update("sellerState", e.target.value)}
                  >
                    {config.states.map((s) => (
                      <option key={s} value={s}>
                        {s} — {config.stateNames[s] ?? s}
                      </option>
                    ))}
                  </select>
                </Field>
                <Field label="Delivering to">
                  <select
                    value={order.customerState}
                    onChange={(e) => update("customerState", e.target.value)}
                  >
                    {config.states.map((s) => (
                      <option key={s} value={s}>
                        {s} — {config.stateNames[s] ?? s}
                      </option>
                    ))}
                  </select>
                </Field>
                <Field label={`Days promised at checkout — ${order.promisedDays}`}>
                  <input
                    type="range"
                    min={1}
                    max={60}
                    value={order.promisedDays}
                    onChange={(e) => update("promisedDays", Number(e.target.value))}
                  />
                </Field>
                {previewDistance !== undefined && (
                  <div style={{ fontSize: 11.5, color: "var(--text-dim)" }}>
                    Typical journey on this route:{" "}
                    <b style={{ color: "var(--accent)" }}>
                      {previewDistance.toLocaleString()} km
                    </b>
                  </div>
                )}
              </div>
            </div>

            {/* ---- the order ---- */}
            <div>
              <div className="kpi-label" style={{ marginBottom: 12 }}>📦 The order</div>
              <div className="grid" style={{ gap: 12 }}>
                <Field label="Product category">
                  <select
                    value={order.category}
                    onChange={(e) => update("category", e.target.value)}
                  >
                    {config.categories.map((c) => (
                      <option key={c} value={c}>{prettify(c)}</option>
                    ))}
                  </select>
                </Field>
                {/* step="any" matters here. With a numeric step such as 10,
                    the browser only accepts values on that ladder (1, 11, 21…)
                    and silently refuses to submit anything in between - which
                    made a perfectly ordinary R$ 480 order "invalid". */}
                <Field label="Order value (R$)">
                  <input
                    type="number" min={1} max={50000} step="any"
                    value={order.orderValue}
                    onChange={(e) => update("orderValue", Number(e.target.value))}
                  />
                </Field>
                <Field label="Freight charged (R$)">
                  <input
                    type="number" min={0} max={2000} step="any"
                    value={order.freightValue}
                    onChange={(e) => update("freightValue", Number(e.target.value))}
                  />
                </Field>
                <Field label="Total weight (grams)">
                  <input
                    type="number" min={1} max={100000} step="any"
                    value={order.weightGrams}
                    onChange={(e) => update("weightGrams", Number(e.target.value))}
                  />
                </Field>
              </div>
            </div>

            {/* ---- basket ---- */}
            <div>
              <div className="kpi-label" style={{ marginBottom: 12 }}>🛒 Basket and payment</div>
              <div className="grid" style={{ gap: 12 }}>
                <Field label="Number of items">
                  <input
                    type="number" min={1} max={50}
                    value={order.itemCount}
                    onChange={(e) => update("itemCount", Number(e.target.value))}
                  />
                </Field>
                <Field label="Number of sellers">
                  <input
                    type="number" min={1} max={10}
                    value={order.sellerCount}
                    onChange={(e) => update("sellerCount", Number(e.target.value))}
                  />
                </Field>
                <Field label="Payment method">
                  <select
                    value={order.paymentType}
                    onChange={(e) => update("paymentType", e.target.value)}
                  >
                    {config.paymentTypes.map((p) => (
                      <option key={p} value={p}>{prettify(p)}</option>
                    ))}
                  </select>
                </Field>
                <Field label="Purchase date">
                  <input
                    type="date"
                    value={order.purchaseDate}
                    onChange={(e) => update("purchaseDate", e.target.value)}
                  />
                </Field>
              </div>
            </div>
          </div>

          <button className="btn" type="submit" disabled={busy} style={{ marginTop: 20, width: "100%" }}>
            {busy ? "Scoring…" : "🔍  Assess delivery risk"}
          </button>
        </form>
      </Glass>

      {error && <ErrorBox message={error} />}

      {/* ---------------- the verdict ---------------- */}
      {result && (
        <div className="fade-in">
          <div className="grid split-2" style={{ marginBottom: 16 }}>
            <Glass>
              <div className="grid split-half" style={{ alignItems: "center", gap: 20 }}>
                <div
                  className="verdict"
                  style={{
                    background: `linear-gradient(145deg, ${colour}, ${colour}bb)`,
                    color: "#06101f",
                  }}
                >
                  <div className="verdict-eyebrow">Prediction</div>
                  <div className="verdict-value">
                    {result.probabilityPct.toFixed(1)}%
                  </div>
                  <div className="verdict-caption">chance of arriving late</div>
                  <div className="verdict-tier">{result.tier} RISK</div>
                </div>

                <div style={{ display: "grid", placeItems: "center" }}>
                  <Gauge
                    value={result.probabilityPct}
                    threshold={result.threshold * 100}
                    colour={colour}
                  />
                  <div style={{ fontSize: 12, color: "var(--text-muted)", textAlign: "center", marginTop: 6 }}>
                    <b style={{ color: colour, fontSize: 17 }}>{result.lift}×</b> the risk of a
                    typical order
                    <div style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 3 }}>
                      marketplace average is {(result.baseRate * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>

              <div className="action-box" style={{ marginTop: 18 }}>
                <b>Recommended action</b>
                {result.action}
              </div>
            </Glass>

            <Glass>
              <SectionHead
                title="What is driving this score?"
                sub={`Journey on this route: ${result.distanceKm.toLocaleString()} km`}
              />
              {result.drivers.map((driver, i) => (
                <div key={i} className={`driver ${driver.level}`}>
                  <span>
                    {driver.level === "high" ? "🔴" : driver.level === "medium" ? "🟡" : "🟢"}
                  </span>
                  <span>{driver.text}</span>
                </div>
              ))}
            </Glass>
          </div>

          <Glass>
            <details>
              <summary
                style={{ cursor: "pointer", fontSize: 13, fontWeight: 620, color: "var(--text-muted)" }}
              >
                See all 34 values sent to the model
              </summary>
              <div
                className="grid"
                style={{
                  gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
                  gap: 8,
                  marginTop: 14,
                }}
              >
                {Object.entries(result.features).map(([key, value]) => (
                  <div
                    key={key}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      gap: 10,
                      fontSize: 11.5,
                      padding: "7px 11px",
                      borderRadius: 9,
                      background: "rgba(255,255,255,0.035)",
                    }}
                  >
                    <span style={{ color: "var(--text-dim)" }}>{key}</span>
                    <b style={{ fontFamily: "var(--font-mono)", color: "var(--text)" }}>
                      {typeof value === "number" ? value.toLocaleString() : value}
                    </b>
                  </div>
                ))}
              </div>
            </details>
          </Glass>
        </div>
      )}
    </div>
  );
}
