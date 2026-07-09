"use client";

import { useState } from "react";

import type { SlateGame } from "@/lib/types";

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
    <div className="grid gap-3 lg:grid-cols-2">
      {slate.map((game) => (
        <GameCard key={game.gameId} game={game} />
      ))}
    </div>
  );
}

function GameCard({ game }: { game: SlateGame }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <article className="rounded-xl border border-[var(--card-border)] bg-[var(--card)] shadow-sm transition hover:border-[var(--accent-muted)]/40">
      <div className="flex items-start justify-between gap-3 px-5 py-4">
        <div className="min-w-0 flex-1">
          <p className="text-base font-semibold leading-snug text-white sm:text-lg">
            {game.awayTeam} @ {game.homeTeam}
          </p>
          <p className="mt-1 text-xs text-[var(--muted)]">
            {game.awayPitcher ?? "TBD"} vs {game.homePitcher ?? "TBD"}
          </p>
        </div>
        <button
          type="button"
          onClick={() => setExpanded((open) => !open)}
          aria-expanded={expanded}
          aria-controls={`lineups-${game.gameId}`}
          className="shrink-0 rounded-lg border border-[var(--card-border)] bg-black/20 px-3 py-1.5 text-xs font-medium text-[var(--accent)] transition hover:border-[var(--accent-muted)] hover:bg-[var(--accent-muted)]/20"
        >
          {expanded ? "Hide Lineups 🔼" : "Show Lineups 🔽"}
        </button>
      </div>

      <div
        id={`lineups-${game.gameId}`}
        className={`grid transition-[grid-template-rows,opacity] duration-300 ease-in-out ${
          expanded ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
        }`}
      >
        <div className="overflow-hidden">
          <div className="grid gap-4 border-t border-[var(--card-border)] px-5 pb-5 pt-4 sm:grid-cols-2">
            <LineupColumn team={game.awayAbbrev} lineup={game.awayLineup} />
            <LineupColumn team={game.homeAbbrev} lineup={game.homeLineup} />
          </div>
        </div>
      </div>
    </article>
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
