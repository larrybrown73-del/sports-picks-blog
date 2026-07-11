import Link from "next/link";

import { WnbaApprovedTable } from "@/components/wnba/WnbaApprovedTable";
import { WnbaGameSection } from "@/components/wnba/WnbaGameSection";
import { formatGeneratedAt } from "@/lib/picks";
import { getLatestWnbaPicks } from "@/lib/wnba/picks";

export const dynamic = "force-dynamic";

export default function WnbaPicksPage() {
  const picks = getLatestWnbaPicks();

  if (!picks) {
    return (
      <div className="space-y-4">
        <h1 className="text-3xl font-bold text-white">Today&apos;s WNBA Picks</h1>
        <p className="rounded-xl border border-[var(--card-border)] bg-[var(--card)] px-5 py-8 text-[var(--muted)]">
          No WNBA pick data found. Run{" "}
          <code>npm run sync-wnba-picks</code> from the blog project root.
        </p>
        <Link href="/" className="text-sm text-[var(--accent)] hover:underline">
          ← Back to MLB picks
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-10">
      <section className="space-y-3">
        <p className="text-sm uppercase tracking-[0.2em] text-[var(--muted)]">
          Daily Slate · {picks.date}
        </p>
        <h1 className="text-3xl font-bold text-white sm:text-4xl">
          Today&apos;s WNBA Picks
        </h1>
        <p className="text-sm text-[var(--muted)]">
          {picks.games.length} games · {picks.approved_count} approved wagers · Run{" "}
          {picks.run_id} · Last synced {formatGeneratedAt(picks.generatedAt)}
        </p>
        <Link href="/" className="inline-block text-sm text-[var(--accent)] hover:underline">
          ← MLB picks
        </Link>
      </section>

      {picks.no_bet ? (
        <div className="rounded-xl border border-amber-900/40 bg-amber-950/20 px-5 py-4 text-sm text-amber-200">
          <p className="font-semibold">No-Bet filter active</p>
          <p className="mt-1 text-amber-100/80">{picks.no_bet_reason}</p>
        </div>
      ) : null}

      {!picks.no_bet && picks.approved_bets.length > 0 ? (
        <WnbaApprovedTable
          bets={picks.approved_bets}
          title={`Slate-Approved Plays (${picks.approved_count})`}
        />
      ) : null}

      <section className="space-y-8">
        <h2 className="text-xl font-semibold text-white">Today&apos;s Slate</h2>
        {picks.games.map((game, index) => (
          <WnbaGameSection key={game.id} game={game} index={index} />
        ))}
      </section>

      <section className="rounded-xl border border-[var(--card-border)] bg-[var(--card)] px-5 py-4 text-xs text-[var(--muted)]">
        Model trained on {picks.models_trained_on.toLocaleString()} completed player-games.
        Spreads use team margin Ridge (σ=11). Props use calibrated PTS/REB/AST projections vs.
        consensus lines. Approved bets pass edge, probability cap, and correlation-adjusted Kelly
        gates.
      </section>
    </div>
  );
}
