import { formatMarketLabel } from "@/lib/utils";
import type { PropPick } from "@/lib/types";

interface PropPickTableProps {
  picks: PropPick[];
  title?: string;
  emptyMessage?: string;
}

export function PropPickTable({
  picks,
  title = "Top Prop Conviction Plays",
  emptyMessage = "No prop edges available for this slate.",
}: PropPickTableProps) {
  return (
    <section className="rounded-xl border border-[var(--card-border)] bg-[var(--card)]">
      <div className="border-b border-[var(--card-border)] px-5 py-4">
        <h2 className="text-lg font-semibold text-white">{title}</h2>
      </div>
      {picks.length === 0 ? (
        <p className="px-5 py-8 text-sm text-[var(--muted)]">{emptyMessage}</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-black/20 text-[var(--muted)]">
              <tr>
                <th className="px-5 py-3 font-medium">Player</th>
                <th className="px-5 py-3 font-medium">Market</th>
                <th className="px-5 py-3 font-medium">Line</th>
                <th className="px-5 py-3 font-medium">Play</th>
                <th className="px-5 py-3 font-medium">Model Prob</th>
                <th className="px-5 py-3 font-medium">Edge</th>
              </tr>
            </thead>
            <tbody>
              {picks.map((pick) => (
                <tr
                  key={`${pick.player}-${pick.market}-${pick.line}`}
                  className="border-t border-[var(--card-border)]"
                >
                  <td className="px-5 py-3 font-medium text-white">{pick.player}</td>
                  <td className="px-5 py-3 text-[var(--muted)]">
                    {formatMarketLabel(pick.market)}
                  </td>
                  <td className="px-5 py-3">{pick.line}</td>
                  <td className="px-5 py-3">{pick.recommendation}</td>
                  <td className="px-5 py-3">
                    {pick.modelProb != null ? `${(pick.modelProb * 100).toFixed(1)}%` : "—"}
                  </td>
                  <td className="px-5 py-3 font-semibold text-[var(--accent)]">
                    {pick.edgePct > 0 ? "+" : ""}
                    {pick.edgePct.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
