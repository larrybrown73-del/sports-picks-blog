import type { PerformanceStats } from "@/lib/types";

interface PerformanceStatsBarProps {
  stats: PerformanceStats;
}

export function PerformanceStatsBar({ stats }: PerformanceStatsBarProps) {
  const items = [
    { label: "Accuracy", value: stats.accuracy },
    { label: "ROI", value: stats.roi },
    { label: "Net Profit", value: stats.netProfit },
    { label: "Brier Score", value: stats.brierScore },
    { label: "Games Scored", value: String(stats.gamesScored) },
    { label: "Bets Placed", value: String(stats.betsPlaced) },
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
        </div>
      ))}
    </section>
  );
}
