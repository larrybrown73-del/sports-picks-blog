import { formatAmericanOdds, formatEdgePct } from "@/lib/formatters";
import type { WnbaApprovedBet } from "@/lib/wnba/types";

interface WnbaApprovedTableProps {
  bets: WnbaApprovedBet[];
  title?: string;
}

function betLabel(bet: WnbaApprovedBet): string {
  if (bet.market === "prop" && bet.player && bet.stat) {
    return `${bet.player} ${bet.direction} ${bet.stat}`;
  }
  return `${bet.team} cover`;
}

export function WnbaApprovedTable({
  bets,
  title = "Platform-Approved Wagers",
}: WnbaApprovedTableProps) {
  const sorted = [...bets].sort((a, b) => b.ev - a.ev);

  return (
    <section className="rounded-xl border border-[var(--card-border)] bg-[var(--card)]">
      <div className="border-b border-[var(--card-border)] px-5 py-4">
        <h2 className="text-lg font-semibold text-white">{title}</h2>
        <p className="mt-1 text-xs text-[var(--muted)]">
          Passed health gates, quality filters, and correlation-adjusted Kelly sizing.
        </p>
      </div>
      {sorted.length === 0 ? (
        <p className="px-5 py-8 text-sm text-[var(--muted)]">
          No approved wagers for this slate.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-black/20 text-[var(--muted)]">
              <tr>
                <th className="px-5 py-3 font-medium">Market</th>
                <th className="px-5 py-3 font-medium">Selection</th>
                <th className="px-5 py-3 font-medium">Line</th>
                <th className="px-5 py-3 font-medium">Odds</th>
                <th className="px-5 py-3 font-medium">Prob</th>
                <th className="px-5 py-3 font-medium">EV</th>
                <th className="px-5 py-3 font-medium">Edge</th>
                <th className="px-5 py-3 font-medium">Kelly</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((bet, index) => (
                <tr
                  key={`${betLabel(bet)}-${bet.line}-${index}`}
                  className="border-t border-[var(--card-border)]"
                >
                  <td className="px-5 py-3 uppercase text-[var(--muted)]">{bet.market}</td>
                  <td className="px-5 py-3 font-medium text-white">{betLabel(bet)}</td>
                  <td className="px-5 py-3">
                    {bet.line != null ? bet.line : "—"}
                  </td>
                  <td className="px-5 py-3">{formatAmericanOdds(bet.odds)}</td>
                  <td className="px-5 py-3">{bet.prob.toFixed(0)}%</td>
                  <td className="px-5 py-3">
                    {bet.ev >= 0 ? "+" : ""}
                    {bet.ev.toFixed(1)}%
                  </td>
                  <td className="px-5 py-3 font-semibold text-[var(--accent)]">
                    {formatEdgePct(bet.edge)}
                  </td>
                  <td className="px-5 py-3">{bet.kelly.toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
