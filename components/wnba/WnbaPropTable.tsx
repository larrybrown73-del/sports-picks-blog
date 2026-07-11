import { formatAmericanOdds, formatEdgePct } from "@/lib/formatters";
import type { WnbaPropLean } from "@/lib/wnba/types";

interface WnbaPropTableProps {
  picks: WnbaPropLean[];
  title?: string;
  emptyMessage?: string;
}

export function WnbaPropTable({
  picks,
  title = "Player Prop Leans",
  emptyMessage = "No prop leans matched for this game.",
}: WnbaPropTableProps) {
  return (
    <section className="rounded-xl border border-[var(--card-border)] bg-[var(--card)]">
      <div className="border-b border-[var(--card-border)] px-5 py-4">
        <h3 className="text-lg font-semibold text-white">{title}</h3>
      </div>
      {picks.length === 0 ? (
        <p className="px-5 py-8 text-sm text-[var(--muted)]">{emptyMessage}</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-black/20 text-[var(--muted)]">
              <tr>
                <th className="px-5 py-3 font-medium">Player</th>
                <th className="px-5 py-3 font-medium">Pick</th>
                <th className="px-5 py-3 font-medium">Line</th>
                <th className="px-5 py-3 font-medium">Proj</th>
                <th className="px-5 py-3 font-medium">Prob</th>
                <th className="px-5 py-3 font-medium">Odds</th>
                <th className="px-5 py-3 font-medium">EV</th>
                <th className="px-5 py-3 font-medium">Edge</th>
                <th className="px-5 py-3 font-medium">Tier</th>
              </tr>
            </thead>
            <tbody>
              {picks.map((pick) => (
                <tr
                  key={`${pick.player}-${pick.stat}-${pick.line}`}
                  className="border-t border-[var(--card-border)]"
                >
                  <td className="px-5 py-3 font-medium text-white">{pick.player}</td>
                  <td className="px-5 py-3">
                    {pick.direction} {pick.stat}
                  </td>
                  <td className="px-5 py-3">{pick.line.toFixed(1)}</td>
                  <td className="px-5 py-3">{pick.projection.toFixed(1)}</td>
                  <td className="px-5 py-3">{pick.prob.toFixed(0)}%</td>
                  <td className="px-5 py-3">{formatAmericanOdds(pick.odds)}</td>
                  <td className="px-5 py-3">
                    {pick.ev >= 0 ? "+" : ""}
                    {pick.ev.toFixed(1)}%
                  </td>
                  <td className="px-5 py-3 font-semibold text-[var(--accent)]">
                    {formatEdgePct(pick.edge)}
                  </td>
                  <td className="px-5 py-3 text-[var(--muted)]">{pick.strength}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
