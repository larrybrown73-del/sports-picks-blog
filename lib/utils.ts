export function formatGeneratedAt(iso: string): string {
  const date = new Date(iso);
  return date.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function formatMarketLabel(market: string): string {
  return market
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export function parsePercent(value: string): number {
  return Number.parseFloat(value.replace(/[%+,]/g, ""));
}
