import type { MoneylinePick } from "@/lib/types";
import {
  confidenceLabelClass,
  edgePctFromPick,
  pickClosingLine,
  pickSelection,
  pickSportsbook,
  pickWinProbability,
  resolveConfidence,
} from "@/lib/confidence";
import { formatAmericanOdds, formatEdgePct } from "@/lib/formatters";

interface PickCardProps {
  pick: MoneylinePick;
  rank: number;
}

export function PickCard({ pick, rank }: PickCardProps) {
  const confidence = resolveConfidence(pick);
  const edgePct = edgePctFromPick(pick);
  const selection = pickSelection(pick);
  const sportsbook = pickSportsbook(pick);
  const winProb = pickWinProbability(pick);

  return (
    <article className="rounded-xl border border-[var(--card-border)] bg-[var(--card)] p-5 shadow-sm transition hover:border-[var(--accent-muted)]">
      <div className="mb-3 flex items-center justify-between">
        <span className="rounded-full bg-[var(--accent-muted)]/40 px-2.5 py-1 text-xs font-medium text-[var(--accent)]">
          #{rank} Value Pick
        </span>
        <span className="text-sm font-semibold text-[var(--accent)]">
          {formatEdgePct(edgePct)} edge
        </span>
      </div>
      <p className="text-lg font-semibold text-white">
        {pick.matchup || `${pick.awayTeam} @ ${pick.homeTeam}`}
      </p>
      <div className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        <div className="rounded-lg bg-black/20 px-3 py-2">
          <p className="text-[var(--muted)]">Play</p>
          <p className="font-medium text-white">
            {selection}
            {sportsbook ? (
              <span className="ml-1 text-xs font-normal text-[var(--muted)]">@ {sportsbook}</span>
            ) : null}
          </p>
        </div>
        <div className="rounded-lg bg-black/20 px-3 py-2">
          <p className="text-[var(--muted)]">Odds</p>
          <p className="font-medium text-white">
            {formatAmericanOdds(pickClosingLine(pick))}
          </p>
        </div>
        <div className="rounded-lg bg-black/20 px-3 py-2">
          <p className="text-[var(--muted)]">Win Prob</p>
          <p className="font-medium text-white">{(winProb * 100).toFixed(1)}%</p>
        </div>
        <div className="rounded-lg bg-black/20 px-3 py-2">
          <p className="text-[var(--muted)]">Confidence</p>
          <p className={`font-medium ${confidenceLabelClass(confidence.label)}`}>
            {pick.confidenceLabel || confidence.label} ({confidence.score})
          </p>
        </div>
      </div>
      {(pick.startingPitcher || pick.weather || pick.umpire || (pick.result && pick.result !== "Pending")) && (
        <p className="mt-3 text-xs text-[var(--muted)]">
          {pick.startingPitcher && (
            <span>
              SP {pick.startingPitcher}
              {pick.opposingPitcher ? ` vs ${pick.opposingPitcher}` : ""}
            </span>
          )}
          {pick.startingPitcher && (pick.weather || pick.umpire || pick.result) && " · "}
          {pick.weather && <span>{pick.weather}</span>}
          {pick.weather && pick.temperature ? ` ${pick.temperature}°F` : null}
          {(pick.weather || pick.temperature) && pick.umpire && " · "}
          {pick.umpire && <span>Ump {pick.umpire}</span>}
          {(pick.startingPitcher || pick.weather || pick.umpire) &&
            pick.result &&
            pick.result !== "Pending" &&
            " · "}
          {pick.result && pick.result !== "Pending" && (
            <span>
              {pick.result} ({pick.profitLoss >= 0 ? "+" : ""}
              {(pick.profitLoss ?? 0).toFixed(2)}u)
            </span>
          )}
        </p>
      )}
    </article>
  );
}
