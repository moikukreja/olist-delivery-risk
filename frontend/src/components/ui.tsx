/**
 * ui.tsx
 * ------
 * Small building blocks reused across every tab.
 *
 * Pulling these out means the glass panel, the KPI tile and the filter pills
 * are defined once. Change the padding here and it changes everywhere, which
 * is what keeps a multi-page app looking like one coherent product.
 */

import type { ReactNode } from "react";

/** A frosted-glass panel. Everything on the page sits inside one of these. */
export function Glass({
  children,
  className = "",
  style,
}: {
  children: ReactNode;
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <div className={`glass glass-pad ${className}`} style={style}>
      {children}
    </div>
  );
}

/** A headline number with a label above it and an optional note below. */
export function Kpi({
  label,
  value,
  note,
  accent,
}: {
  label: string;
  value: string;
  note?: string;
  accent?: string;
}) {
  return (
    <Glass>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value" style={accent ? { color: accent } : undefined}>
        {value}
      </div>
      {note && <div className="kpi-note">{note}</div>}
    </Glass>
  );
}

/** A titled section header used above every chart. */
export function SectionHead({ title, sub }: { title: string; sub?: string }) {
  return (
    <>
      <h2 className="section-title">{title}</h2>
      {sub && <p className="section-sub">{sub}</p>}
    </>
  );
}

/**
 * A set of toggle pills used instead of a multi-select dropdown.
 *
 * Native multi-selects are genuinely awkward - you have to hold Ctrl and
 * click, and you cannot see what is chosen without scrolling. Pills show every
 * selection at a glance and take one click each.
 */
export function PillGroup({
  options,
  selected,
  onToggle,
  labelFor,
}: {
  options: string[];
  selected: string[];
  onToggle: (value: string) => void;
  labelFor?: (value: string) => string;
}) {
  return (
    <div className="pill-group">
      {options.map((option) => (
        <button
          key={option}
          type="button"
          className={`pill ${selected.includes(option) ? "on" : ""}`}
          onClick={() => onToggle(option)}
        >
          {labelFor ? labelFor(option) : option}
        </button>
      ))}
    </div>
  );
}

/** A labelled form control. */
export function Field({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="field">
      <label>{label}</label>
      {children}
    </div>
  );
}

export function Loading({ message }: { message: string }) {
  return (
    <div className="loading">
      <div className="spinner" />
      <div>{message}</div>
    </div>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <Glass style={{ borderColor: "rgba(255,92,122,0.4)" }}>
      <div style={{ color: "var(--risk-veryhigh)", fontWeight: 650, marginBottom: 6 }}>
        Something went wrong
      </div>
      <div style={{ fontSize: 12.5, color: "var(--text-muted)", lineHeight: 1.6 }}>
        {message}
      </div>
    </Glass>
  );
}

/**
 * Colour that matches a risk tier.
 *
 * These are real hex values, not `var(--risk-high)`. That matters: the verdict
 * card builds a gradient by appending an alpha suffix to the colour, and
 * "var(--risk-high)bb" is not valid CSS - it silently fails and the card loses
 * its fill. A hex string concatenates correctly.
 */
export function tierColour(tier: string): string {
  switch (tier) {
    case "VERY HIGH":
      return "#ff5c7a";
    case "HIGH":
      return "#ff9f45";
    case "MODERATE":
      return "#ffd166";
    default:
      return "#2ee6a8";
  }
}

/** Turn 1234567 into "1.2M" so KPI tiles stay readable. */
export function compact(value: number): string {
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(0)}K`;
  return value.toFixed(0);
}

/** "health_beauty" -> "Health beauty" for display in charts and dropdowns. */
export function prettify(value: string): string {
  const spaced = value.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}
