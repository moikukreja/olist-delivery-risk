/**
 * BrazilMap.tsx
 * -------------
 * A WebGL choropleth of Brazil's 27 states, drawn with deck.gl.
 *
 * WHAT "CHOROPLETH" MEANS
 * A map where each region is shaded according to a number. Here, the redder a
 * state is, the more of its deliveries arrive late.
 *
 * WHY deck.gl AND NOT A NORMAL CHART LIBRARY
 * deck.gl pushes the map geometry onto the graphics card and draws it with
 * WebGL, the same technology that powers 3D games. A normal SVG chart asks the
 * browser to manage thousands of individual shapes, which gets sluggish. Here
 * the GPU redraws all 27 states plus the volume bubbles in well under a
 * millisecond, so hovering and filtering feel instant.
 *
 * There is deliberately NO online map underneath. Every byte the map needs -
 * including the state outlines - is bundled into the app, so it works with no
 * external requests, no API keys and no tile-server bills.
 */

import { useMemo, useState } from "react";
import DeckGL from "@deck.gl/react";
import { GeoJsonLayer, ScatterplotLayer } from "@deck.gl/layers";
import type { Color, PickingInfo } from "@deck.gl/core";
import type { Feature, FeatureCollection, Geometry } from "geojson";
import type { StatePoint } from "../types";
import brazilStates from "../assets/brazil-states.json";

/** The two fields our simplified GeoJSON carries on each state. */
interface StateProperties {
  code: string;
  name: string;
}

type StateFeature = Feature<Geometry, StateProperties>;

// The imported JSON is just a plain object as far as TypeScript is concerned,
// so we tell it once, here, what shape that object really has.
const BRAZIL = brazilStates as unknown as FeatureCollection<Geometry, StateProperties>;

// Centred on Brazil's geographic middle. The zoom is deliberately modest so
// the whole country - including the far south and the Amazon north - fits
// inside the panel without the user having to pan.
const INITIAL_VIEW = {
  longitude: -54,
  latitude: -14,
  zoom: 2.7,
  minZoom: 2.2,
  maxZoom: 7,
  pitch: 0,
  bearing: 0,
};

/** Colour ramp from calm teal (safe) through amber to hot pink (bad). */
const RAMP: [number, number, number][] = [
  [46, 230, 168],
  [120, 220, 190],
  [255, 209, 102],
  [255, 159, 69],
  [255, 92, 122],
];

/**
 * Turn a late-delivery percentage into an RGB colour.
 *
 * `max` is the worst rate currently on screen, so the full colour range is
 * always used no matter how the user has filtered the data.
 */
function colourFor(rate: number, max: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, rate / Math.max(max, 0.001)));
  const scaled = t * (RAMP.length - 1);
  const index = Math.min(RAMP.length - 2, Math.floor(scaled));
  const blend = scaled - index;
  const from = RAMP[index];
  const to = RAMP[index + 1];
  return [
    Math.round(from[0] + (to[0] - from[0]) * blend),
    Math.round(from[1] + (to[1] - from[1]) * blend),
    Math.round(from[2] + (to[2] - from[2]) * blend),
  ];
}

interface Props {
  states: StatePoint[];
}

export default function BrazilMap({ states }: Props) {
  const [hovered, setHovered] = useState<{
    state: StatePoint;
    x: number;
    y: number;
  } | null>(null);

  // Build a quick code -> data lookup so the map layer can colour each polygon
  // without searching the whole array 27 times over.
  const { byCode, worstRate } = useMemo(() => {
    const lookup = new Map<string, StatePoint>();
    let worst = 0;
    for (const state of states) {
      lookup.set(state.code, state);
      if (state.lateRate > worst) worst = state.lateRate;
    }
    return { byCode: lookup, worstRate: worst };
  }, [states]);

  const maxOrders = useMemo(
    () => Math.max(...states.map((s) => s.orders), 1),
    [states]
  );

  const layers = [
    // Layer 1: the filled state shapes, coloured by late rate.
    new GeoJsonLayer<StateProperties>({
      id: "states",
      data: BRAZIL,
      filled: true,
      stroked: true,
      pickable: true,
      getFillColor: (feature: StateFeature): Color => {
        const state = byCode.get(feature.properties.code);
        if (!state) return [255, 255, 255, 16];   // no data for this state
        const [r, g, b] = colourFor(state.lateRate, worstRate);
        return [r, g, b, 165];
      },
      getLineColor: [255, 255, 255, 60] as Color,
      getLineWidth: 1,
      lineWidthUnits: "pixels",
      updateTriggers: {
        getFillColor: [byCode, worstRate],
      },
      onHover: (info: PickingInfo<StateFeature>) => {
        const state = info.object ? byCode.get(info.object.properties.code) : undefined;
        setHovered(state ? { state, x: info.x, y: info.y } : null);
      },
    }),

    // Layer 2: a bubble per state whose SIZE shows order volume. Colour tells
    // you how bad it is; size tells you how much it matters.
    new ScatterplotLayer<StatePoint>({
      id: "volume",
      data: states,
      pickable: true,
      radiusUnits: "pixels",
      getPosition: (d: StatePoint) => [d.lng, d.lat],
      getRadius: (d: StatePoint) => 4 + Math.sqrt(d.orders / maxOrders) * 26,
      getFillColor: [255, 255, 255, 40] as Color,
      getLineColor: [255, 255, 255, 190] as Color,
      getLineWidth: 1.4,
      lineWidthUnits: "pixels",
      stroked: true,
      updateTriggers: { getRadius: [maxOrders] },
      onHover: (info: PickingInfo<StatePoint>) => {
        setHovered(info.object ? { state: info.object, x: info.x, y: info.y } : null);
      },
    }),
  ];

  return (
    <div className="map-wrap">
      <DeckGL
        initialViewState={INITIAL_VIEW}
        controller={{ dragRotate: false }}
        layers={layers}
        style={{ position: "absolute", inset: "0" }}
        getCursor={() => (hovered ? "pointer" : "grab")}
      />

      {hovered && (
        <div
          className="map-tooltip"
          style={{
            // Nudge the tooltip away from the cursor, and flip it to the left
            // near the right-hand edge so it never runs off the panel.
            left: hovered.x > 320 ? hovered.x - 190 : hovered.x + 16,
            top: hovered.y + 14,
          }}
        >
          <div style={{ fontWeight: 700, marginBottom: 3 }}>
            {hovered.state.name} ({hovered.state.code})
          </div>
          <div>
            Late rate:{" "}
            <b
              style={{
                color: `rgb(${colourFor(hovered.state.lateRate, worstRate).join(",")})`,
              }}
            >
              {hovered.state.lateRate.toFixed(2)}%
            </b>
          </div>
          <div>Orders: {hovered.state.orders.toLocaleString()}</div>
          <div>Typical journey: {hovered.state.medianDistance.toLocaleString()} km</div>
          <div>Revenue: R$ {(hovered.state.revenue / 1000).toFixed(0)}K</div>
        </div>
      )}

      <div className="map-legend">
        <div style={{ fontWeight: 650, color: "var(--text)" }}>
          Late-delivery rate
        </div>
        <div className="legend-scale">
          {RAMP.map((c, i) => (
            <div
              key={i}
              style={{ flex: 1, background: `rgb(${c.join(",")})` }}
            />
          ))}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span>0%</span>
          <span>{worstRate.toFixed(1)}%</span>
        </div>
        <div style={{ marginTop: 6, opacity: 0.75 }}>
          Circle size = order volume · drag to pan, scroll to zoom
        </div>
      </div>
    </div>
  );
}
