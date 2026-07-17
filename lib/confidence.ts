import type { MoneylinePick } from "@/lib/types";

export type ConfidenceLabel = "Low" | "Medium" | "High" | "Elite";

function normalizeConfidenceLabel(value: string | undefined): ConfidenceLabel {
  const label = (value || "").trim();
  if (label === "Elite" || label === "High" || label === "Medium" || label === "Low") {
    return label;
  }
  return "Low";
}

/** Edge percentage points (e.g. 7.4) from export edge or legacy fields. */
export function edgePctFromPick(pick: MoneylinePick): number {
  if (typeof pick.edge === "number" && !Number.isNaN(pick.edge)) {
    const edge = pick.edge;
    if (Math.abs(edge) <= 1 && pick.winProbability != null) {
      return edge * 100;
    }
    return edge;
  }
  if (typeof pick.edgeDecimal === "number" && !Number.isNaN(pick.edgeDecimal)) {
    return pick.edgeDecimal * 100;
  }
  if (typeof pick.edgePct === "number" && !Number.isNaN(pick.edgePct)) {
    return pick.edgePct;
  }
  return 0;
}

export function pickSelection(pick: MoneylinePick): string {
  return pick.pick || pick.play || "";
}

export function pickSportsbook(pick: MoneylinePick): string {
  return pick.sportsbook || pick.book || "";
}

export function pickClosingLine(pick: MoneylinePick): number | undefined {
  if (typeof pick.closingLine === "number") return pick.closingLine;
  if (typeof pick.openingLine === "number") return pick.openingLine;
  if (typeof pick.americanOdds === "number") return pick.americanOdds;
  return undefined;
}

export function pickWinProbability(pick: MoneylinePick): number {
  if (typeof pick.winProbability === "number") return pick.winProbability;
  if (typeof pick.modelWinProb === "number") return pick.modelWinProb;
  return 0;
}

// Label thresholds align with export Low/Medium/High buckets on edge %.
export function confidenceFromEdge(edgePct: number, _modelWinProb = 0.52): {
  score: number;
  label: ConfidenceLabel;
} {
  let label: ConfidenceLabel = "Low";
  if (edgePct >= 5.0) {
    label = "High";
  } else if (edgePct >= 3.0) {
    label = "Medium";
  }
  const score = Math.min(100, Math.max(0, Math.round(edgePct * 4.0 + 40)));
  return { score, label };
}

export function resolveConfidence(pick: MoneylinePick): {
  score: number;
  label: ConfidenceLabel;
} {
  if (typeof pick.confidence === "number" && !Number.isNaN(pick.confidence)) {
    const score = Math.round(pick.confidence);
    if (pick.confidenceLabel) {
      return { score, label: normalizeConfidenceLabel(pick.confidenceLabel) };
    }
    if (score >= 85) return { score, label: "Elite" };
    if (score >= 70) return { score, label: "High" };
    if (score >= 50) return { score, label: "Medium" };
    return { score, label: "Low" };
  }
  if (typeof pick.confidence === "string" && pick.confidence.trim()) {
    const label = normalizeConfidenceLabel(pick.confidence);
    const edgePct = edgePctFromPick(pick);
    return { score: Math.min(100, Math.max(0, Math.round(edgePct * 4.0 + 40))), label };
  }
  if (pick.confidenceScore != null && pick.confidenceLabel) {
    return { score: pick.confidenceScore, label: pick.confidenceLabel };
  }
  return confidenceFromEdge(edgePctFromPick(pick), pickWinProbability(pick) || 0.52);
}

export function confidenceLabelClass(label: ConfidenceLabel): string {
  switch (label) {
    case "Elite":
      return "text-emerald-300";
    case "High":
      return "text-[var(--accent)]";
    case "Medium":
      return "text-amber-300";
    default:
      return "text-[var(--muted)]";
  }
}
