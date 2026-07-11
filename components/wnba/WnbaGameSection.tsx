import { WnbaApprovedTable } from "@/components/wnba/WnbaApprovedTable";
import { WnbaProjectionsTable } from "@/components/wnba/WnbaProjectionsTable";
import { WnbaPropTable } from "@/components/wnba/WnbaPropTable";
import { WnbaSpreadCard } from "@/components/wnba/WnbaSpreadCard";
import { formatWnbaTip } from "@/lib/wnba/picks";
import type { WnbaGame } from "@/lib/wnba/types";

interface WnbaGameSectionProps {
  game: WnbaGame;
  index: number;
}

export function WnbaGameSection({ game, index }: WnbaGameSectionProps) {
  const awayPlayers = game.teams[game.away] ?? [];
  const homePlayers = game.teams[game.home] ?? [];
  const spreadPicks = [...game.spreads].sort((a, b) => b.ev - a.ev);
  const propPicks = [...game.props].sort((a, b) => b.ev - a.ev).slice(0, 12);

  return (
    <section className="space-y-6 rounded-2xl border border-[var(--card-border)] bg-black/10 p-5 sm:p-6">
      <div className="space-y-1">
        <p className="text-xs uppercase tracking-[0.2em] text-[var(--muted)]">
          Game {index + 1}
        </p>
        <h2 className="text-2xl font-bold text-white">
          {game.away} @ {game.home}
        </h2>
        <p className="text-sm text-[var(--muted)]">
          {game.date} · {formatWnbaTip(game.tip)} ET · {game.status}
        </p>
      </div>

      {game.approved.length > 0 ? (
        <WnbaApprovedTable
          bets={game.approved}
          title={`Approved Plays · ${game.away.split(" ").pop()} @ ${game.home.split(" ").pop()}`}
        />
      ) : null}

      {spreadPicks.length > 0 ? (
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-white">Spreads</h3>
          <div className="grid gap-4 md:grid-cols-2">
            {spreadPicks.map((pick, rank) => (
              <WnbaSpreadCard key={pick.team} pick={pick} rank={rank + 1} />
            ))}
          </div>
        </div>
      ) : null}

      <WnbaPropTable picks={propPicks} />

      <div className="space-y-3">
        <h3 className="text-lg font-semibold text-white">Player Projections</h3>
        <div className="grid gap-4 lg:grid-cols-2">
          <WnbaProjectionsTable team={game.away} players={awayPlayers} />
          <WnbaProjectionsTable team={game.home} players={homePlayers} />
        </div>
      </div>
    </section>
  );
}
