import type { DailyPicks, SlateGame } from "@/lib/types";

interface SlateBoardProps {
  slate: SlateGame[];
}

export function SlateBoard({ slate }: SlateBoardProps) {
  if (slate.length === 0) {
    return (
      <p className="rounded-xl border border-[var(--card-border)] bg-[var(--card)] px-5 py-8 text-sm text-[var(--muted)]">
        No slate data available for this date.
      </p>
    );
  }

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {slate.map((game) => (
        <article
          key={game.gameId}
          className="rounded-xl border border-[var(--card-border)] bg-[var(--card)] p-5"
        >
          <header className="mb-4 border-b border-[var(--card-border)] pb-3">
            <p className="text-lg font-semibold text-white">
              {game.awayTeam} @ {game.homeTeam}
            </p>
            <p className="mt-1 text-xs text-[var(--muted)]">
              {game.awayPitcher ?? "TBD"} vs {game.homePitcher ?? "TBD"}
            </p>
          </header>
          <div className="grid gap-4 sm:grid-cols-2">
            <LineupColumn team={game.awayAbbrev} lineup={game.awayLineup} />
            <LineupColumn team={game.homeAbbrev} lineup={game.homeLineup} />
          </div>
        </article>
      ))}
    </div>
  );
}

function LineupColumn({
  team,
  lineup,
}: {
  team: string;
  lineup: SlateGame["awayLineup"];
}) {
  return (
    <div>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">
        {team} Lineup
      </h3>
      <ol className="space-y-1 text-sm">
        {lineup.map((player) => (
          <li key={`${team}-${player.playerId}`} className="flex gap-2 text-[var(--foreground)]">
            <span className="w-4 text-[var(--muted)]">{player.slot}.</span>
            <span>{player.name}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}
