/**
 * App.tsx
 * -------
 * The shell: background, header, tab navigation, and whichever tab is open.
 *
 * The app fetches its configuration once on startup. Until that arrives we
 * show a spinner; if it fails we show why, rather than a blank screen.
 */

import { useEffect, useState } from "react";
import { fetchConfig } from "./api";
import type { AppConfig } from "./types";
import About from "./components/About";
import BatchUpload from "./components/BatchUpload";
import Dashboard from "./components/Dashboard";
import ModelInsights from "./components/ModelInsights";
import Predict from "./components/Predict";
import ShaderBackground from "./components/ShaderBackground";
import { ErrorBox, Loading } from "./components/ui";

const TABS = [
  { key: "dashboard", label: "📊  Dashboard" },
  { key: "predict", label: "🔮  Predict an Order" },
  { key: "batch", label: "📦  Batch Upload" },
  { key: "model", label: "📈  Model Insights" },
  { key: "about", label: "ℹ️  About" },
] as const;

type TabKey = (typeof TABS)[number]["key"];

export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<TabKey>("dashboard");

  useEffect(() => {
    fetchConfig()
      .then(setConfig)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <>
      <ShaderBackground />

      <div className="app-shell">
        <header className="header">
          <div className="brand">
            <div className="brand-mark">📦</div>
            <div>
              <h1>Olist Delivery Risk Radar</h1>
              <p>
                Predicting late deliveries on Brazil&rsquo;s largest e-commerce
                marketplace
              </p>
            </div>
          </div>

          {config && (
            <div className="header-stats">
              <span className="chip">
                <b>{config.reference.total_orders.toLocaleString()}</b> orders analysed
              </span>
              <span className="chip">
                <b>{config.model.name}</b> · ROC-AUC{" "}
                <b>{config.model.metrics["ROC-AUC"].toFixed(3)}</b>
              </span>
              <span className="chip">
                Base late rate <b>{(config.model.baseRate * 100).toFixed(2)}%</b>
              </span>
              {/* This chip is the visible proof of automated deployment. It was
                  added AFTER the Space went live, pushed to GitHub, and reached
                  the live site without anyone touching Hugging Face. */}
              <span className="chip chip-live">
                <span className="live-dot" /> Auto-deployed by GitHub Actions
              </span>
            </div>
          )}
        </header>

        <nav className="tabs glass">
          {TABS.map((t) => (
            <button
              key={t.key}
              className={`tab ${tab === t.key ? "active" : ""}`}
              onClick={() => setTab(t.key)}
            >
              {t.label}
            </button>
          ))}
        </nav>

        {error && <ErrorBox message={error} />}
        {!config && !error && <Loading message="Loading the model and 96,470 orders…" />}

        {config && (
          <main>
            {tab === "dashboard" && <Dashboard config={config} />}
            {tab === "predict" && <Predict config={config} />}
            {tab === "batch" && <BatchUpload />}
            {tab === "model" && <ModelInsights config={config} />}
            {tab === "about" && <About config={config} />}
          </main>
        )}

        <footer className="footer-note">
          Software Development for AI Models · Final Group Project · Prof. JP Agarwal
          <br />
          React + deck.gl WebGL frontend · FastAPI + scikit-learn backend · deployed to
          Hugging Face Spaces by GitHub Actions
        </footer>
      </div>
    </>
  );
}
