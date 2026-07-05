export function formatCurrency(value: number | string): string {
  const numeric =
    typeof value === "number"
      ? value
      : Number.parseFloat(String(value).replace(/[$,+]/g, ""));

  if (Number.isNaN(numeric)) {
    return String(value);
  }

  const sign = numeric >= 0 ? "+" : "-";
  return `${sign}$${Math.abs(numeric).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

export function formatPercent(value: number): string {
  return `${value.toFixed(1)}%`;
}

export function formatSignedPercent(value: string | number): string {
  if (typeof value === "number") {
    const sign = value >= 0 ? "+" : "";
    return `${sign}${value.toFixed(2)}%`;
  }

  const trimmed = value.trim();
  if (trimmed.startsWith("+") || trimmed.startsWith("-")) {
    return trimmed;
  }
  return trimmed;
}

export function formatEdgePct(value: number): string {
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;
}

export function formatAmericanOdds(value: number | undefined | null): string {
  if (value == null || value === 0) {
    return "—";
  }
  return value > 0 ? `+${value}` : String(value);
}

export function parseNetProfit(value: string): number {
  return Number.parseFloat(value.replace(/[$,+]/g, ""));
}
