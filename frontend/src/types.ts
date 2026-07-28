/**
 * types.ts
 * --------
 * TypeScript descriptions of every piece of JSON the Python backend sends us.
 *
 * WHY BOTHER?
 * JavaScript will happily let you write `data.lateRate` when the field is
 * actually called `late_rate`, and you only find out when the page breaks in
 * front of your audience. TypeScript checks these shapes while you are still
 * writing the code, so that whole class of mistake becomes impossible.
 */

export interface StateCoordinate {
  lat: number;
  lng: number;
}

export interface ModelMetrics {
  Accuracy: number;
  Precision: number;
  Recall: number;
  F1: number;
  "ROC-AUC": number;
  "PR-AUC": number;
  Brier?: number;
}

export interface AppConfig {
  states: string[];
  stateNames: Record<string, string>;
  stateCoordinates: Record<string, StateCoordinate>;
  routeDistances: Record<string, number>;
  categories: string[];
  paymentTypes: string[];
  months: string[];
  reference: {
    overall_late_rate: number;
    total_orders: number;
    median_distance_km: number;
    median_order_value: number;
    median_freight: number;
    median_estimated_days: number;
    median_weight_g: number;
    date_range: [string, string];
  };
  model: {
    name: string;
    calibration: string;
    trainedAt: string;
    threshold: number;
    baseRate: number;
    trainingRows: number;
    testRows: number;
    sklearnVersion: string;
    metrics: ModelMetrics;
    comparison: Record<string, ModelMetrics>;
    topFeatures: { feature: string; importance: number }[];
  };
}

export interface BandPoint {
  band: string;
  orders: number;
  lateRate: number;
}

export interface StatePoint {
  code: string;
  name: string;
  lat: number;
  lng: number;
  orders: number;
  lateRate: number;
  revenue: number;
  medianDistance: number;
}

export interface DashboardData {
  fellBackToAllData: boolean;
  kpis: {
    orders: number;
    shareOfAllOrders: number;
    lateRate: number;
    lateOrders: number;
    revenue: number;
    revenueAtRisk: number;
    revenueAtRiskPct: number;
    medianDistance: number;
    medianDeliveryDays: number;
  };
  monthly: { month: string; orders: number; lateRate: number }[];
  states: StatePoint[];
  categories: { name: string; orders: number; lateRate: number }[];
  distanceBands: BandPoint[];
  promiseBands: BandPoint[];
  reviews:
    | { available: false }
    | {
        available: true;
        byScore: { score: number; onTime: number; late: number }[];
        avgOnTime: number;
        avgLate: number;
        poorOnTime: number;
        poorLate: number;
      };
}

export interface OrderRequest {
  sellerState: string;
  customerState: string;
  promisedDays: number;
  category: string;
  orderValue: number;
  freightValue: number;
  weightGrams: number;
  itemCount: number;
  sellerCount: number;
  paymentType: string;
  installments: number;
  purchaseDate: string;
}

export interface PredictionResult {
  probability: number;
  probabilityPct: number;
  tier: "LOW" | "MODERATE" | "HIGH" | "VERY HIGH";
  action: string;
  lift: number;
  threshold: number;
  baseRate: number;
  distanceKm: number;
  flagged: boolean;
  drivers: { level: "low" | "medium" | "high"; text: string }[];
  features: Record<string, number | string>;
}

export interface BatchSummary {
  rowsSubmitted: number;
  rowsScored: number;
  rowsRejected: number;
  flagged: number;
  flaggedPct: number;
  meanProbabilityPct: number;
  baseRatePct: number;
  tiers: Record<"LOW" | "MODERATE" | "HIGH" | "VERY HIGH", number>;
  revenueFlagged?: number;
  revenueTotal?: number;
}

export interface BatchResult {
  summary: BatchSummary;
  errors: { row: number; error: string }[];
  columns: string[];
  preview: Record<string, string | number>[];
  /** The complete scored file as CSV text, ready to offer as a download. */
  csv: string;
  filename: string;
}

export interface DashboardFilters {
  monthFrom: string | null;
  monthTo: string | null;
  states: string[];
  categories: string[];
  payments: string[];
}
