/**
 * Dashboard.tsx
 * -------------
 * Interactive analytics across all 96,470 historical orders.
 *
 * Every filter the user touches triggers one request to the Python backend,
 * which does the pandas grouping and sends back a few hundred numbers. We
 * deliberately do NOT ship the raw 96,470 rows to the browser: that would be a
 * 3 MB download and would make filtering sluggish on a phone.
 */

import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchDashboard } from "../api";
import type { AppConfig, DashboardData, DashboardFilters } from "../types";
import BrazilMap from "./BrazilMap";
import { ErrorBox, Field, Glass, Kpi, Loading, PillGroup, SectionHead, compact, prettify } from "./ui";

const AXIS = { stroke: "#737d9e", fontSize: 11 };
const GRID = "rgba(255,255,255,0.06)";

const TOOLTIP_STYLE = {
  background: "rgba(6,10,28,0.95)",
  border: "1px solid rgba(255,255,255,0.2)",
  borderRadius: 12,
  fontSize: 12,
  color: "#eef2ff",
};

/**
 * A category label for the vertical bar chart.
 *
 * Recharts' built-in tick wraps long text onto two lines, which turned
 * "construction_tools_construction" into a messy stack. This renders one
 * single line and trims anything too long with an ellipsis instead.
 */
function CategoryTick({
  x = 0,
  y = 0,
  payload,
}: {
  x?: number;
  y?: number;
  payload?: { value: string };
}) {
  const raw = prettify(payload?.value ?? "");
  const label = raw.length > 16 ? `${raw.slice(0, 15)}…` : raw;
  return (
    <text x={x} y={y} dy={4} textAnchor="end" fill="#737d9e" fontSize={10.5}>
      {label}
    </text>
  );
}

/** Shade a bar by how bad its value is, using the same ramp as the map. */
function riskColour(rate: number, max: number): string {
  const t = Math.max(0, Math.min(1, rate / Math.max(max, 0.001)));
  if (t < 0.3) return "#2ee6a8";
  if (t < 0.55) return "#ffd166";
  if (t < 0.8) return "#ff9f45";
  return "#ff5c7a";
}

interface Props {
  config: AppConfig;
}

export default function Dashboard({ config }: Props) {
  const months = config.months;

  const [filters, setFilters] = useState<DashboardFilters>({
    monthFrom: months[0] ?? null,
    monthTo: months[months.length - 1] ?? null,
    states: [],
    categories: [],
    payments: [],
  });

  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setBusy(true);
    fetchDashboard(filters)
      .then((result) => {
        // If the user changed the filters while this request was still in
        // flight, throw the stale answer away rather than flashing it on screen.
        if (!cancelled) {
          setData(result);
          setError(null);
        }
      })
      .catch((e: Error) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setBusy(false));
    return () => {
      cancelled = true;
    };
  }, [filters]);

  function toggle(key: "states" | "categories" | "payments", value: string) {
    setFilters((f) => ({
      ...f,
      [key]: f[key].includes(value)
        ? f[key].filter((v) => v !== value)
        : [...f[key], value],
    }));
  }

  const worstCategoryRate = useMemo(
    () => Math.max(...(data?.categories.map((c) => c.lateRate) ?? [1])),
    [data]
  );

  if (error) return <ErrorBox message={error} />;
  if (!data) return <Loading message="Crunching 96,470 orders..." />;

  const { kpis } = data;

  return (
    <div className="fade-in">
      {/* ---------------- filters ---------------- */}
      <Glass style={{ marginBottom: 16 }}>
        <SectionHead
          title="Filter the marketplace"
          sub="Every chart below responds instantly. Leave a filter empty to include everything."
        />
        <div className="filter-bar">
          <Field label="From month">
            <select
              value={filters.monthFrom ?? ""}
              onChange={(e) => setFilters((f) => ({ ...f, monthFrom: e.target.value }))}
            >
              {months.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </Field>
          <Field label="To month">
            <select
              value={filters.monthTo ?? ""}
              onChange={(e) => setFilters((f) => ({ ...f, monthTo: e.target.value }))}
            >
              {months.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </Field>
          <Field label={`Destination state${filters.states.length ? ` (${filters.states.length})` : ""}`}>
            <PillGroup
              options={config.states}
              selected={filters.states}
              onToggle={(v) => toggle("states", v)}
            />
          </Field>
          <Field label={`Payment method${filters.payments.length ? ` (${filters.payments.length})` : ""}`}>
            <PillGroup
              options={config.paymentTypes}
              selected={filters.payments}
              onToggle={(v) => toggle("payments", v)}
              labelFor={prettify}
            />
          </Field>
        </div>

        {(filters.states.length > 0 ||
          filters.categories.length > 0 ||
          filters.payments.length > 0) && (
          <button
            className="btn-ghost"
            style={{ marginTop: 12 }}
            onClick={() =>
              setFilters((f) => ({ ...f, states: [], categories: [], payments: [] }))
            }
          >
            Clear all filters
          </button>
        )}

        {data.fellBackToAllData && (
          <div className="callout" style={{ marginTop: 12 }}>
            No orders match that combination, so the charts below show{" "}
            <b>all orders</b> instead. Try widening your selection.
          </div>
        )}
      </Glass>

      {/* ---------------- KPI tiles ---------------- */}
      <div className="grid kpi-row" style={{ opacity: busy ? 0.55 : 1, transition: "opacity .2s" }}>
        <Kpi
          label="Orders"
          value={kpis.orders.toLocaleString()}
          note={`${kpis.shareOfAllOrders}% of the marketplace`}
        />
        <Kpi
          label="Late-delivery rate"
          value={`${kpis.lateRate}%`}
          note={`${kpis.lateOrders.toLocaleString()} orders arrived late`}
          accent={riskColour(kpis.lateRate, 20)}
        />
        <Kpi label="Revenue" value={`R$ ${compact(kpis.revenue)}`} note="Total order value" />
        <Kpi
          label="Revenue delivered late"
          value={`R$ ${compact(kpis.revenueAtRisk)}`}
          note={`${kpis.revenueAtRiskPct}% of revenue`}
          accent="var(--risk-high)"
        />
        <Kpi
          label="Typical journey"
          value={`${kpis.medianDistance.toLocaleString()} km`}
          note={`Median delivery takes ${kpis.medianDeliveryDays} days`}
        />
      </div>

      {/* ---------------- trend + map ---------------- */}
      <div className="grid split-2" style={{ marginBottom: 16 }}>
        <Glass>
          <SectionHead
            title="Order volume and late rate over time"
            sub="Bars are order volume; the line is the percentage delivered late."
          />
          <ResponsiveContainer width="100%" height={400}>
            <ComposedChart data={data.monthly} margin={{ top: 6, right: 8, left: -14, bottom: 0 }}>
              <defs>
                <linearGradient id="volumeFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#4f7cff" stopOpacity={0.85} />
                  <stop offset="100%" stopColor="#4f7cff" stopOpacity={0.25} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="month" {...AXIS} tickLine={false} />
              <YAxis yAxisId="left" {...AXIS} tickLine={false} axisLine={false} />
              <YAxis
                yAxisId="right"
                orientation="right"
                {...AXIS}
                tickLine={false}
                axisLine={false}
                unit="%"
              />
              <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
              <Legend wrapperStyle={{ fontSize: 11.5, paddingTop: 6 }} />
              <Bar yAxisId="left" dataKey="orders" name="Orders" fill="url(#volumeFill)" radius={[4, 4, 0, 0]} />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="lateRate"
                name="Late rate (%)"
                stroke="#ff5c7a"
                strokeWidth={2.6}
                dot={{ r: 2.5, fill: "#ff5c7a" }}
                activeDot={{ r: 5 }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </Glass>

        <Glass>
          <SectionHead
            title="Where deliveries fail"
            sub="Rendered on the GPU with deck.gl — hover any state for detail."
          />
          <BrazilMap states={data.states} />
        </Glass>
      </div>

      {/* ---------------- three drivers ---------------- */}
      <div className="grid split-3" style={{ marginBottom: 16 }}>
        <Glass>
          <SectionHead title="Distance drives risk" sub="Seller to buyer, in kilometres" />
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data.distanceBands} margin={{ top: 6, right: 6, left: -22, bottom: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="band" {...AXIS} tickLine={false} />
              <YAxis {...AXIS} tickLine={false} axisLine={false} unit="%" />
              <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
              <Bar dataKey="lateRate" name="Late rate (%)" radius={[5, 5, 0, 0]}>
                {data.distanceBands.map((d, i) => (
                  <Cell key={i} fill={riskColour(d.lateRate, 14)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Glass>

        <Glass>
          <SectionHead
            title="A tight promise is a broken promise"
            sub="Days pledged to the customer at checkout"
          />
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data.promiseBands} margin={{ top: 6, right: 6, left: -22, bottom: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="band" {...AXIS} tickLine={false} />
              <YAxis {...AXIS} tickLine={false} axisLine={false} unit="%" />
              <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
              <Bar dataKey="lateRate" name="Late rate (%)" radius={[5, 5, 0, 0]}>
                {data.promiseBands.map((d, i) => (
                  <Cell key={i} fill={riskColour(d.lateRate, 32)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Glass>

        <Glass>
          <SectionHead title="Riskiest product categories" sub="Categories with 40+ orders" />
          <ResponsiveContainer width="100%" height={280}>
            <BarChart
              layout="vertical"
              data={[...data.categories].reverse()}
              margin={{ top: 4, right: 12, left: 52, bottom: 0 }}
            >
              <CartesianGrid stroke={GRID} horizontal={false} />
              <XAxis type="number" {...AXIS} tickLine={false} axisLine={false} unit="%" />
              <YAxis
                type="category"
                dataKey="name"
                width={112}
                tickLine={false}
                axisLine={false}
                tick={<CategoryTick />}
                // Without interval={0} Recharts thins the labels out when it
                // thinks they are crowded, leaving half the bars unnamed.
                interval={0}
              />
              <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
              <Bar dataKey="lateRate" name="Late rate (%)" radius={[0, 5, 5, 0]}>
                {[...data.categories].reverse().map((d, i) => (
                  <Cell key={i} fill={riskColour(d.lateRate, worstCategoryRate)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Glass>
      </div>

      {/* ---------------- commercial impact ---------------- */}
      {data.reviews.available && (
        <div className="grid split-2">
          <Glass>
            <SectionHead
              title="Why this matters commercially"
              sub="Distribution of customer review scores, split by whether the order arrived on time"
            />
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data.reviews.byScore} margin={{ top: 6, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid stroke={GRID} vertical={false} />
                <XAxis dataKey="score" {...AXIS} tickLine={false} unit="★" />
                <YAxis {...AXIS} tickLine={false} axisLine={false} unit="%" />
                <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
                <Legend wrapperStyle={{ fontSize: 11.5, paddingTop: 6 }} />
                <Bar dataKey="onTime" name="Delivered on time" fill="#2ee6a8" radius={[5, 5, 0, 0]} />
                <Bar dataKey="late" name="Delivered late" fill="#ff5c7a" radius={[5, 5, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Glass>

          <Glass>
            <SectionHead title="The cost of being late" sub="Averaged over the current selection" />
            <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div>
                <div className="kpi-label">Stars when on time</div>
                <div className="kpi-value" style={{ color: "var(--risk-low)" }}>
                  {data.reviews.avgOnTime.toFixed(2)}★
                </div>
              </div>
              <div>
                <div className="kpi-label">Stars when late</div>
                <div className="kpi-value" style={{ color: "var(--risk-veryhigh)" }}>
                  {data.reviews.avgLate.toFixed(2)}★
                </div>
              </div>
            </div>

            <div style={{ marginTop: 22 }}>
              <div className="kpi-label">Share of 1–2 star reviews</div>
              <ResponsiveContainer width="100%" height={130}>
                <BarChart
                  layout="vertical"
                  data={[
                    { label: "On time", value: data.reviews.poorOnTime, fill: "#2ee6a8" },
                    { label: "Late", value: data.reviews.poorLate, fill: "#ff5c7a" },
                  ]}
                  margin={{ top: 8, right: 40, left: 8, bottom: 0 }}
                >
                  <XAxis type="number" hide domain={[0, 100]} />
                  <YAxis type="category" dataKey="label" {...AXIS} width={62} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
                  <Bar dataKey="value" name="1–2 star reviews (%)" radius={[0, 6, 6, 0]}>
                    <Cell fill="#2ee6a8" />
                    <Cell fill="#ff5c7a" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="action-box" style={{ marginTop: 8 }}>
              <b>The business case</b>
              A late delivery makes a 1–2 star review{" "}
              <b style={{ color: "var(--text)" }}>
                {(data.reviews.poorLate / Math.max(data.reviews.poorOnTime, 0.01)).toFixed(1)}×
              </b>{" "}
              more likely. Every late delivery predicted in advance is a customer
              relationship there is still time to protect.
            </div>
          </Glass>
        </div>
      )}
    </div>
  );
}
