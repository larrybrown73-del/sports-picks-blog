import type { MoneylinePick } from "@/lib/types";
import { confidenceLabelClass, resolveConfidence } from "@/lib/confidence";
import { formatAmericanOdds, formatEdgePct } from "@/lib/formatters";

interface PickCardProps {
  pick: MoneylinePick;
  rank: number;
}

export function PickCard({ pick, rank }: PickCardProps) {
  const confidence = resolveConfidence(pick);

  return (
    <article className="rounded-xl border border-[var(--card-border)] bg-[var(--card)] p-5 shadow-sm transition hover:border-[var(--accent-muted)]">
      <div className="mb-3 flex items-center justify-between">
        <span className="rounded-full bg-[var(--accent-muted)]/40 px-2.5 py-1 text-xs font-medium text-[var(--accent)]">
          #{rank} Value Pick
        </span>
        <span className="text-sm font-semibold text-[var(--accent)]">
          {formatEdgePct(pick.edgePct)} edge
        </span>
      </div>
      <p className="text-lg font-semibold text-white">
        {pick.awayTeam} @ {pick.homeTeam}
      </p>
      <div className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        <div className="rounded-lg bg-black/20 px-3 py-2">
          <p className="text-[var(--muted)]">Play</p>
          <p className="font-medium text-white">
            {pick.play}
            {pick.book ? (
              <span className="ml-1 text-xs font-normal text-[var(--muted)]">@ {pick.book}</span>
            ) : null}
          </p>
        </div>
        <div className="rounded-lg bg-black/20 px-3 py-2">
          <p className="text-[var(--muted)]">Odds</p>
          <p className="font-medium text-white">{formatAmericanOdds(pick.americanOdds)}</p>
        </div>
        <div className="rounded-lg bg-black/20 px-3 py-2">
          <p className="text-[var(--muted)]">Quarter-Kelly</p>
          <p className="font-medium text-white">{pick.sizingPct.toFixed(2)}%</p>
        </div>
        <div className="rounded-lg bg-black/20 px-3 py-2">
          <p className="text-[var(--muted)]">Confidence</p>
          <p className={`font-medium ${confidenceLabelClass(confidence.label)}`}>
            {pick.confidenceTier ?? confidence.label}{" "}
            <span className="text-white/70">({confidence.score})</span>
          </p>
        </div>
      </div>
      {(pick.evPerUnit != null || pick.confidenceTier || pick.predHomeRuns != null) && (
        <p className="mt-3 text-xs text-[var(--muted)]">
          {pick.evPerUnit != null && (
            <span>EV {pick.evPerUnit >= 0 ? "+" : ""}{pick.evPerUnit.toFixed(3)} / unit</span>
          )}
          {pick.evPerUnit != null && (pick.confidenceTier || pick.predHomeRuns != null) && " · "}
          {pick.confidenceTier && <span>{pick.confidenceTier}</span>}
          {pick.confidenceTier && pick.predHomeRuns != null && " · "}
          {pick.predHomeRuns != null && pick.predAwayRuns != null && (
            <span>
              Proj {pick.awayTeam.split(" ").pop()} {pick.predAwayRuns} – {pick.homeTeam.split(" ").pop()}{" "}
              {pick.predHomeRuns}
            </span>
          )}
        </p>
      )}
    </article>
  );
}
