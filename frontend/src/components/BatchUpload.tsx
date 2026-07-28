/**
 * BatchUpload.tsx
 * ---------------
 * Score a whole CSV of orders in one go.
 *
 * WHY THIS EXISTS
 * The single-order form suits a call-centre agent checking one customer. It is
 * useless to an operations team facing this morning's five thousand orders.
 * Batch scoring turns the model from a lookup tool into something that can be
 * run against a full day of trading before any of it ships.
 *
 * The same twelve fields the form collects become the twelve CSV columns, so
 * anyone who understands the form already understands the file format.
 */

import { useCallback, useRef, useState } from "react";
import { BATCH_TEMPLATE_URL, predictBatch } from "../api";
import type { BatchResult } from "../types";
import { ErrorBox, Glass, Kpi, SectionHead, compact, prettify, tierColour } from "./ui";

const REQUIRED_COLUMNS: [string, string][] = [
  ["seller_state", "Two-letter state code the parcel ships from, e.g. SP"],
  ["customer_state", "Two-letter state code the parcel ships to, e.g. AM"],
  ["promised_days", "Days promised to the customer at checkout"],
  ["category", "Product category, e.g. health_beauty"],
  ["order_value", "Order value in Brazilian reais"],
  ["freight_value", "Freight charged in Brazilian reais"],
  ["weight_grams", "Total order weight in grams"],
  ["item_count", "Number of items in the order"],
  ["seller_count", "Number of distinct sellers in the order"],
  ["payment_type", "Payment method, e.g. credit_card"],
  ["installments", "Number of payment instalments"],
  ["purchase_date", "Date the order was placed, YYYY-MM-DD"],
];

const TIER_ORDER = ["LOW", "MODERATE", "HIGH", "VERY HIGH"] as const;

export default function BatchUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<BatchResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const choose = useCallback((chosen: File | null) => {
    setFile(chosen);
    setResult(null);
    setError(null);
  }, []);

  function onDrop(event: React.DragEvent) {
    event.preventDefault();
    setDragging(false);
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) choose(dropped);
  }

  async function run() {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await predictBatch(file));
    } catch (e) {
      setError((e as Error).message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  /**
   * Turn the returned CSV text into a file the browser downloads.
   *
   * The server already sent the complete results, so building the download in
   * memory avoids a second round trip and works even if the Space goes to
   * sleep immediately afterwards.
   */
  function download() {
    if (!result) return;
    const blob = new Blob([result.csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = result.filename;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="fade-in">
      {/* ---------------- upload ---------------- */}
      <Glass style={{ marginBottom: 16 }}>
        <SectionHead
          title="Score a whole file of orders"
          sub="Upload a CSV and every row is scored in one pass. A 2,000-order file takes well under a second."
        />

        <div
          className={`dropzone ${dragging ? "over" : ""} ${file ? "has-file" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".csv,text/csv"
            style={{ display: "none" }}
            onChange={(e) => choose(e.target.files?.[0] ?? null)}
          />
          <div style={{ fontSize: 34, marginBottom: 8 }}>{file ? "📄" : "📥"}</div>
          {file ? (
            <>
              <div style={{ fontWeight: 680, fontSize: 14 }}>{file.name}</div>
              <div style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 4 }}>
                {(file.size / 1024).toFixed(0)} KB · click to choose a different file
              </div>
            </>
          ) : (
            <>
              <div style={{ fontWeight: 650, fontSize: 14 }}>
                Drop a CSV here, or click to browse
              </div>
              <div style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 4 }}>
                Up to 20,000 orders per file
              </div>
            </>
          )}
        </div>

        <div style={{ display: "flex", gap: 10, marginTop: 16, flexWrap: "wrap" }}>
          <button className="btn" onClick={run} disabled={!file || busy}>
            {busy ? "Scoring…" : "🚀  Score all orders"}
          </button>
          <a className="btn-ghost" href={BATCH_TEMPLATE_URL} download
             style={{ textDecoration: "none", display: "inline-flex", alignItems: "center" }}>
            ⬇  Download template CSV
          </a>
          {file && (
            <button className="btn-ghost" onClick={() => choose(null)}>
              Clear
            </button>
          )}
        </div>

        <details style={{ marginTop: 16 }}>
          <summary style={{ cursor: "pointer", fontSize: 13, fontWeight: 620, color: "var(--text-muted)" }}>
            Required columns ({REQUIRED_COLUMNS.length})
          </summary>
          <div style={{ overflowX: "auto", marginTop: 12 }}>
            <table className="table">
              <thead>
                <tr>
                  <th>Column</th>
                  <th>Meaning</th>
                </tr>
              </thead>
              <tbody>
                {REQUIRED_COLUMNS.map(([name, meaning]) => (
                  <tr key={name}>
                    <td style={{ fontFamily: "var(--font-mono)", fontSize: 11.5 }}>{name}</td>
                    <td>{meaning}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="section-sub" style={{ marginTop: 10, marginBottom: 0 }}>
            Any <code>order_id</code>, <code>customer_id</code>, <code>reference</code> or{" "}
            <code>notes</code> column is carried straight through to the results so you can
            join the scores back to your own records. Extra columns are ignored.
          </p>
        </details>
      </Glass>

      {error && <ErrorBox message={error} />}

      {/* ---------------- results ---------------- */}
      {result && (
        <div className="fade-in">
          <div className="grid kpi-row">
            <Kpi
              label="Orders scored"
              value={result.summary.rowsScored.toLocaleString()}
              note={
                result.summary.rowsRejected > 0
                  ? `${result.summary.rowsRejected} row(s) rejected`
                  : "every row accepted"
              }
            />
            <Kpi
              label="Flagged for action"
              value={result.summary.flagged.toLocaleString()}
              note={`${result.summary.flaggedPct}% of the file`}
              accent="var(--risk-high)"
            />
            <Kpi
              label="Average risk"
              value={`${result.summary.meanProbabilityPct}%`}
              note={`marketplace average is ${result.summary.baseRatePct}%`}
            />
            {result.summary.revenueFlagged !== undefined && (
              <Kpi
                label="Revenue flagged"
                value={`R$ ${compact(result.summary.revenueFlagged)}`}
                note={`of R$ ${compact(result.summary.revenueTotal ?? 0)} total`}
                accent="var(--risk-veryhigh)"
              />
            )}
            <Kpi
              label="Very high risk"
              value={result.summary.tiers["VERY HIGH"].toLocaleString()}
              note="need action before dispatch"
              accent="var(--risk-veryhigh)"
            />
          </div>

          {/* risk mix */}
          <Glass style={{ marginBottom: 16 }}>
            <SectionHead title="Risk breakdown" sub="Every scored order placed into one of four bands" />
            <div className="tier-bar">
              {TIER_ORDER.map((tier) => {
                const count = result.summary.tiers[tier];
                const share = (count / Math.max(result.summary.rowsScored, 1)) * 100;
                if (share <= 0) return null;
                return (
                  <div
                    key={tier}
                    className="tier-seg"
                    style={{ width: `${share}%`, background: tierColour(tier) }}
                    title={`${tier}: ${count.toLocaleString()} (${share.toFixed(1)}%)`}
                  />
                );
              })}
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 18, marginTop: 12 }}>
              {TIER_ORDER.map((tier) => (
                <div key={tier} style={{ display: "flex", alignItems: "center", gap: 7 }}>
                  <span
                    style={{
                      width: 11, height: 11, borderRadius: 3,
                      background: tierColour(tier), display: "inline-block",
                    }}
                  />
                  <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
                    {tier}{" "}
                    <b style={{ color: "var(--text)" }}>
                      {result.summary.tiers[tier].toLocaleString()}
                    </b>
                  </span>
                </div>
              ))}
            </div>
          </Glass>

          {/* rejected rows */}
          {result.errors.length > 0 && (
            <Glass style={{ marginBottom: 16, borderColor: "rgba(255,159,69,0.35)" }}>
              <SectionHead
                title={`${result.summary.rowsRejected} row(s) could not be scored`}
                sub="The rest of the file was processed normally. Row numbers match the line in your CSV."
              />
              <div style={{ overflowX: "auto", maxHeight: 220 }}>
                <table className="table">
                  <thead>
                    <tr>
                      <th style={{ width: 80 }}>Row</th>
                      <th>Problem</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.errors.map((e) => (
                      <tr key={e.row}>
                        <td className="num">{e.row}</td>
                        <td>{e.error}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Glass>
          )}

          {/* preview + download */}
          <Glass>
            <div style={{ display: "flex", justifyContent: "space-between",
                          alignItems: "flex-start", gap: 16, flexWrap: "wrap" }}>
              <div>
                <SectionHead
                  title="Results"
                  sub={`Showing the first ${Math.min(result.preview.length, 100)} of ${result.summary.rowsScored.toLocaleString()} scored orders. The download contains every row and every column.`}
                />
              </div>
              <button className="btn" onClick={download}>
                ⬇  Download scored CSV
              </button>
            </div>

            <div style={{ overflowX: "auto", maxHeight: 520 }}>
              <table className="table">
                <thead>
                  <tr>
                    {result.columns.map((c) => (
                      <th key={c} style={{ whiteSpace: "nowrap" }}>
                        {prettify(c)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.preview.map((row, i) => (
                    <tr key={i}>
                      {result.columns.map((c) => {
                        const value = row[c];
                        if (c === "risk_tier") {
                          return (
                            <td key={c}>
                              <span
                                className="tier-pill"
                                style={{
                                  background: `${tierColour(String(value))}22`,
                                  color: tierColour(String(value)),
                                  borderColor: `${tierColour(String(value))}55`,
                                }}
                              >
                                {value}
                              </span>
                            </td>
                          );
                        }
                        if (c === "late_probability_pct") {
                          return (
                            <td key={c} className="num" style={{
                              color: tierColour(String(row["risk_tier"])),
                              fontWeight: 650,
                            }}>
                              {Number(value).toFixed(1)}%
                            </td>
                          );
                        }
                        return (
                          <td key={c} className={typeof value === "number" ? "num" : ""}
                              style={{ whiteSpace: "nowrap" }}>
                            {typeof value === "number" ? value.toLocaleString() : String(value)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Glass>
        </div>
      )}

      {/* ---------------- API documentation ---------------- */}
      {!result && (
        <Glass>
          <SectionHead
            title="Calling this from your own code"
            sub="The same endpoint is a public API, so batch scoring can run from a scheduled job rather than a browser."
          />
          <div className="prose">
            <p>
              Every endpoint is documented interactively at{" "}
              <a href="/docs" target="_blank" rel="noreferrer">
                /docs
              </a>
              , generated automatically from the server code so it can never fall out
              of date. From Python:
            </p>
          </div>
          <pre className="code-block">{`import requests

with open("orders.csv", "rb") as fh:
    response = requests.post(
        "https://samuelalex37-olist-delivery-risk.hf.space/api/predict/batch",
        files={"file": fh},
    )

result = response.json()
print(result["summary"])

with open("scored.csv", "w", encoding="utf-8") as out:
    out.write(result["csv"])`}</pre>
        </Glass>
      )}
    </div>
  );
}
