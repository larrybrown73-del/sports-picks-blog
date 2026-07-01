import type { PerformanceStats } from "@/lib/types";
import { formatCurrency, formatSignedPercent, parseNetProfit } from "@/lib/formatters";

interface PerformanceStatsBarProps {
  stats: PerformanceStats;
}

export function PerformanceStatsBar({ stats }: PerformanceStatsBarProps) {
  const netProfitNumeric = parseNetProfit(stats.netProfit);

  const items = [
    { label: "Accuracy", value: formatSignedPercent(stats.accuracy) },
    { label: "ROI", value: formatSignedPercent(stats.roi) },
    {
      label: "Net Profit (simulated)",
      value: formatCurrency(netProfitNumeric),
      hint: "Based on $1,000 starting bankroll backtest",
    },
    { label: "Brier Score", value: stats.brierScore },
    { label: "Games Scored", value: stats.gamesScored.toLocaleString("en-US") },
    { label: "Bets Placed", value: stats.betsPlaced.toLocaleString("en-US") },
  ];

  return (
    <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {items.map((item) => (
        <div
          key={item.label}
          className="rounded-xl border border-[var(--card-border)] bg-[var(--card)] px-4 py-3"
        >
          <p className="text-xs uppercase tracking-wide text-[var(--muted)]">
            {item.label}
          </p>
          <p className="mt-1 text-lg font-semibold text-white">{item.value}</p>
          {"hint" in item && item.hint ? (
            <p className="mt-1 text-xs text-[var(--muted)]">{item.hint}</p>
          ) : null}
        </div>
      ))}
    </section>
  );
}
