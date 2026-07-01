import { formatAmericanOdds, formatCurrency } from "@/lib/formatters";
import { getYesterdayResults } from "@/lib/results";
import type { GradedPickResult } from "@/lib/types";

interface YesterdayResultsTableProps {
  todayDate?: string;
}

function resultClass(result: GradedPickResult): string {
  switch (result) {
    case "Win":
      return "text-[var(--accent)] font-semibold";
    case "Loss":
      return "text-red-400 font-semibold";
    default:
      return "text-[var(--muted)]";
  }
}

export function YesterdayResultsTable({ todayDate }: YesterdayResultsTableProps) {
  const results = getYesterdayResults(todayDate);

  if (!results || results.picks.length === 0) {
    return (
      <section className="rounded-xl border border-[var(--card-border)] bg-[var(--card)]">
        <div className="border-b border-[var(--card-border)] px-5 py-4">
          <h2 className="text-lg font-semibold text-white">Yesterday&apos;s Results</h2>
        </div>
        <p className="px-5 py-8 text-sm text-[var(--muted)]">
          No graded picks yet. Run <code>npm run grade-picks</code> after games finish.
        </p>
      </section>
    );
  }

  return (
    <section className="rounded-xl border border-[var(--card-border)] bg-[var(--card)]">
      <div className="border-b border-[var(--card-border)] px-5 py-4">
        <h2 className="text-lg font-semibold text-white">
          Yesterday&apos;s Results ({results.date})
        </h2>
        <p className="mt-1 text-sm text-[var(--muted)]">
          {results.summary.wins}-{results.summary.losses} record
          {results.summary.pending > 0 ? ` · ${results.summary.pending} pending` : ""}
          {" · "}
          {formatCurrency(results.summary.netProfitLoss)} on ${results.unitStake} units
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-sm">
          <thead className="bg-black/20 text-[var(--muted)]">
            <tr>
              <th className="px-5 py-3 font-medium">Date</th>
              <th className="px-5 py-3 font-medium">Pick</th>
              <th className="px-5 py-3 font-medium">Odds</th>
              <th className="px-5 py-3 font-medium">Result</th>
              <th className="px-5 py-3 font-medium">Profit/Loss</th>
            </tr>
          </thead>
          <tbody>
            {results.picks.map((row) => (
              <tr key={`${row.date}-${row.matchup}-${row.pick}`} className="border-t border-[var(--card-border)]">
                <td className="px-5 py-3">{row.date}</td>
                <td className="px-5 py-3">
                  <p className="font-medium text-white">{row.pick}</p>
                  <p className="text-xs text-[var(--muted)]">{row.matchup}</p>
                </td>
                <td className="px-5 py-3">{formatAmericanOdds(row.americanOdds)}</td>
                <td className={`px-5 py-3 ${resultClass(row.result)}`}>{row.result}</td>
                <td className="px-5 py-3">{formatCurrency(row.profitLoss)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="border-t border-[var(--card-border)] px-5 py-3 text-xs text-[var(--muted)]">
        Odds are model-estimated lines at pick time. P/L uses flat ${results.unitStake} unit stakes.
      </p>
    </section>
  );
}
