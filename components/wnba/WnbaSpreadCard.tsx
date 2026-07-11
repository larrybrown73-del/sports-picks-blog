import { formatAmericanOdds, formatEdgePct } from "@/lib/formatters";
import type { WnbaSpreadPick } from "@/lib/wnba/types";

interface WnbaSpreadCardProps {
  pick: WnbaSpreadPick;
  rank: number;
}

export function WnbaSpreadCard({ pick, rank }: WnbaSpreadCardProps) {
  const lineLabel = pick.spread > 0 ? `+${pick.spread}` : String(pick.spread);

  return (
    <article className="rounded-xl border border-[var(--card-border)] bg-[var(--card)] p-5 shadow-sm transition hover:border-[var(--accent-muted)]">
      <div className="mb-3 flex items-center justify-between">
        <span className="rounded-full bg-[var(--accent-muted)]/40 px-2.5 py-1 text-xs font-medium text-[var(--accent)]">
          #{rank} Spread
        </span>
        <span className="text-sm font-semibold text-[var(--accent)]">
          {formatEdgePct(pick.edge)} edge
        </span>
      </div>
      <p className="text-lg font-semibold text-white">{pick.team}</p>
      <div className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        <div className="rounded-lg bg-black/20 px-3 py-2">
          <p className="text-[var(--muted)]">Line</p>
          <p className="font-medium text-white">{lineLabel}</p>
        </div>
        <div className="rounded-lg bg-black/20 px-3 py-2">
          <p className="text-[var(--muted)]">Odds</p>
          <p className="font-medium text-white">{formatAmericanOdds(pick.odds)}</p>
        </div>
        <div className="rounded-lg bg-black/20 px-3 py-2">
          <p className="text-[var(--muted)]">Cover Prob</p>
          <p className="font-medium text-white">{pick.coverProb.toFixed(1)}%</p>
        </div>
        <div className="rounded-lg bg-black/20 px-3 py-2">
          <p className="text-[var(--muted)]">Kelly</p>
          <p className="font-medium text-white">{pick.kelly.toFixed(2)}%</p>
        </div>
      </div>
      <p className="mt-3 text-xs text-[var(--muted)]">
        EV {pick.ev >= 0 ? "+" : ""}
        {pick.ev.toFixed(1)}% · Model margin {pick.margin >= 0 ? "+" : ""}
        {pick.margin.toFixed(1)}
      </p>
    </article>
  );
}
