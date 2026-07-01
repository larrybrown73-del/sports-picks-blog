import Link from "next/link";

import { DailyPicksTabs } from "@/components/DailyPicksTabs";
import { PerformanceStatsBar } from "@/components/PerformanceStatsBar";
import { PropPickTable } from "@/components/PropPickTable";
import { PickCard } from "@/components/PickCard";
import { SlateBoard } from "@/components/SlateBoard";
import { formatGeneratedAt, getLatestPicks, getPickDates } from "@/lib/picks";

export const dynamic = "force-dynamic";

export default function HomePage() {
  const picks = getLatestPicks();
  const archiveDates = getPickDates();

  if (!picks) {
    return (
      <div className="space-y-4">
        <h1 className="text-3xl font-bold text-white">Today&apos;s MLB Picks</h1>
        <p className="rounded-xl border border-[var(--card-border)] bg-[var(--card)] px-5 py-8 text-[var(--muted)]">
          No pick data found. Add JSON files under <code>data/picks/</code> or run{" "}
          <code>npm run sync-picks</code>.
        </p>
      </div>
    );
  }

  const topProps = picks.propPicks.conviction.slice(0, 10);
  const slate = picks.slate ?? [];

  return (
    <div className="space-y-10">
      <section className="space-y-3">
        <p className="text-sm uppercase tracking-[0.2em] text-[var(--muted)]">
          Daily Slate · {picks.date}
        </p>
        <h1 className="text-3xl font-bold text-white sm:text-4xl">
          Today&apos;s MLB Picks
        </h1>
        <p className="text-sm text-[var(--muted)]">
          Last synced {formatGeneratedAt(picks.generatedAt)}
        </p>
        {archiveDates.length > 1 && (
          <Link
            href={`/picks/${picks.date}`}
            className="inline-block text-sm text-[var(--accent)] hover:underline"
          >
            View full slate archive →
          </Link>
        )}
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-white">System Performance</h2>
        <PerformanceStatsBar stats={picks.performance} />
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-white">
          Lineup &amp; Slate Details
          {slate.length > 0 && (
            <span className="ml-2 text-base font-normal text-[var(--muted)]">
              ({slate.length} games)
            </span>
          )}
        </h2>
        <SlateBoard slate={slate} />
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-white">Moneyline Value Picks</h2>
        {picks.moneylinePicks.length === 0 ? (
          <p className="rounded-xl border border-[var(--card-border)] bg-[var(--card)] px-5 py-8 text-sm text-[var(--muted)]">
            No games met the 3% implied market edge threshold today.
          </p>
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {picks.moneylinePicks.map((pick, index) => (
              <PickCard key={`${pick.awayTeam}-${pick.homeTeam}`} pick={pick} rank={index + 1} />
            ))}
          </div>
        )}
      </section>

      <section className="space-y-4">
        <PropPickTable
          picks={topProps}
          title="Top Prop Conviction Plays"
          emptyMessage="No prop conviction plays available for this slate."
        />
      </section>

      <section className="space-y-4 border-t border-[var(--card-border)] pt-8">
        <h2 className="text-xl font-semibold text-white">Full Slate</h2>
        <DailyPicksTabs picks={picks} />
      </section>
    </div>
  );
}
