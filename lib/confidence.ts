import type { MoneylinePick } from "@/lib/types";

export function confidenceFromEdge(edgePct: number, modelWinProb = 0.52): {
  score: number;
  label: MoneylinePick["confidenceLabel"];
} {
  const score = Math.min(
    100,
    Math.max(0, Math.round(edgePct * 4.0 + (modelWinProb - 0.5) * 80)),
  );
  let label: NonNullable<MoneylinePick["confidenceLabel"]> = "Low";
  if (score >= 85) {
    label = "Elite";
  } else if (score >= 70) {
    label = "High";
  } else if (score >= 50) {
    label = "Medium";
  }
  return { score, label };
}

export function resolveConfidence(pick: MoneylinePick): {
  score: number;
  label: NonNullable<MoneylinePick["confidenceLabel"]>;
} {
  if (pick.confidenceScore != null && pick.confidenceLabel) {
    return { score: pick.confidenceScore, label: pick.confidenceLabel };
  }
  return confidenceFromEdge(pick.edgePct, pick.modelWinProb ?? 0.52);
}

export function confidenceLabelClass(
  label: NonNullable<MoneylinePick["confidenceLabel"]>,
): string {
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
