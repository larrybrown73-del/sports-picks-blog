import { ModelNotes } from "@/components/ModelNotes";
import { PerformanceChart } from "@/components/PerformanceChart";
import { PerformanceStatsBar } from "@/components/PerformanceStatsBar";
import { formatCurrency, formatSignedPercent, parseNetProfit } from "@/lib/formatters";
import { getAllPerformanceSnapshots, getLatestPicks } from "@/lib/picks";

export const dynamic = "force-dynamic";

export default function PerformancePage() {
  const snapshots = getAllPerformanceSnapshots();
  const latest = getLatestPicks();

  return (
    <div className="space-y-10">
      <section className="space-y-3">
        <p className="text-sm uppercase tracking-[0.2em] text-[var(--muted)]">
          Model Health
        </p>
        <h1 className="text-3xl font-bold text-white sm:text-4xl">
          System Performance
        </h1>
        <p className="max-w-2xl text-[var(--muted)]">
          Historical backtest metrics exported from the baseball-predictor performance log.
          Each snapshot reflects the system state at pick export time.
        </p>
      </section>

      <ModelNotes />

      {latest ? (
        <section className="space-y-4">
          <h2 className="text-xl font-semibold text-white">Latest Snapshot</h2>
          <PerformanceStatsBar stats={latest.performance} />
        </section>
      ) : null}

      <PerformanceChart snapshots={snapshots} />

      <section className="rounded-xl border border-[var(--card-border)] bg-[var(--card)]">
        <div className="border-b border-[var(--card-border)] px-5 py-4">
          <h2 className="text-lg font-semibold text-white">Performance History</h2>
        </div>
        {snapshots.length === 0 ? (
          <p className="px-5 py-8 text-sm text-[var(--muted)]">
            No snapshots yet. Run <code>npm run sync-picks</code> to export daily metrics.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-black/20 text-[var(--muted)]">
                <tr>
                  <th className="px-5 py-3 font-medium">Date</th>
                  <th className="px-5 py-3 font-medium">Accuracy</th>
                  <th className="px-5 py-3 font-medium">ROI</th>
                  <th className="px-5 py-3 font-medium">Net Profit</th>
                  <th className="px-5 py-3 font-medium">Brier</th>
                  <th className="px-5 py-3 font-medium">Games</th>
                  <th className="px-5 py-3 font-medium">Bets</th>
                </tr>
              </thead>
              <tbody>
                {snapshots.map((snapshot) => (
                  <tr
                    key={`${snapshot.date}-${snapshot.generatedAt}`}
                    className="border-t border-[var(--card-border)]"
                  >
                    <td className="px-5 py-3 font-medium text-white">{snapshot.date}</td>
                    <td className="px-5 py-3">{formatSignedPercent(snapshot.performance.accuracy)}</td>
                    <td className="px-5 py-3 text-[var(--accent)]">
                      {formatSignedPercent(snapshot.performance.roi)}
                    </td>
                    <td className="px-5 py-3">
                      {formatCurrency(parseNetProfit(snapshot.performance.netProfit))}
                    </td>
                    <td className="px-5 py-3">{snapshot.performance.brierScore}</td>
                    <td className="px-5 py-3">
                      {snapshot.performance.gamesScored.toLocaleString("en-US")}
                    </td>
                    <td className="px-5 py-3">
                      {snapshot.performance.betsPlaced.toLocaleString("en-US")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
