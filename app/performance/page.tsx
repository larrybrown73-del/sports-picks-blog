import { PerformanceChart } from "@/components/PerformanceChart";
import { getAllPerformanceSnapshots } from "@/lib/picks";

export default function PerformancePage() {
  const snapshots = getAllPerformanceSnapshots();

  return (
    <div className="space-y-8">
      <section className="space-y-3">
        <h1 className="text-3xl font-bold text-white">System Performance</h1>
        <p className="max-w-2xl text-[var(--muted)]">
          Historical backtest metrics exported from the baseball-predictor performance log.
          Each snapshot reflects the system state at pick export time.
        </p>
      </section>

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
                    <td className="px-5 py-3">{snapshot.performance.accuracy}</td>
                    <td className="px-5 py-3 text-[var(--accent)]">
                      {snapshot.performance.roi}
                    </td>
                    <td className="px-5 py-3">{snapshot.performance.netProfit}</td>
                    <td className="px-5 py-3">{snapshot.performance.brierScore}</td>
                    <td className="px-5 py-3">{snapshot.performance.gamesScored}</td>
                    <td className="px-5 py-3">{snapshot.performance.betsPlaced}</td>
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
