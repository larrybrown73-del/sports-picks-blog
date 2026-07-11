import type { WnbaPlayerProjection } from "@/lib/wnba/types";

interface WnbaProjectionsTableProps {
  team: string;
  players: WnbaPlayerProjection[];
}

export function WnbaProjectionsTable({ team, players }: WnbaProjectionsTableProps) {
  return (
    <section className="rounded-xl border border-[var(--card-border)] bg-[var(--card)]">
      <div className="border-b border-[var(--card-border)] px-5 py-4">
        <h3 className="text-lg font-semibold text-white">{team}</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-sm">
          <thead className="bg-black/20 text-[var(--muted)]">
            <tr>
              <th className="px-5 py-3 font-medium">Player</th>
              <th className="px-5 py-3 font-medium">PTS</th>
              <th className="px-5 py-3 font-medium">REB</th>
              <th className="px-5 py-3 font-medium">AST</th>
              <th className="px-5 py-3 font-medium">PRA</th>
            </tr>
          </thead>
          <tbody>
            {players.map((player) => (
              <tr
                key={player.name}
                className="border-t border-[var(--card-border)]"
              >
                <td className="px-5 py-3 font-medium text-white">
                  {player.name}{" "}
                  <span className="text-[var(--muted)]">({player.pos})</span>
                </td>
                <td className="px-5 py-3">{player.pts.toFixed(1)}</td>
                <td className="px-5 py-3">{player.reb.toFixed(1)}</td>
                <td className="px-5 py-3">{player.ast.toFixed(1)}</td>
                <td className="px-5 py-3 font-semibold text-[var(--accent)]">
                  {player.pra.toFixed(1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
