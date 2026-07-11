import Link from "next/link";

import { PropPickTable } from "@/components/PropPickTable";
import { PickCard } from "@/components/PickCard";
import { SlateBoard } from "@/components/SlateBoard";
import { YesterdayResultsTable } from "@/components/YesterdayResultsTable";
import { formatGeneratedAt, getLatestPicks, getPickDates } from "@/lib/picks";
import { resolvePickCounts, slateHeaderLabel } from "@/lib/pickMeta";

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
  const { slateGames, scheduleGames } = resolvePickCounts(picks);
  const propsAvailable = picks.meta?.propsAvailable ?? topProps.length > 0;
  const slateMissing = scheduleGames != null && scheduleGames > 0 && slateGames === 0;

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
          {slateHeaderLabel(picks)} · Last synced {formatGeneratedAt(picks.generatedAt)}
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

      <YesterdayResultsTable todayDate={picks.date} />

      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-white">Today&apos;s Slate &amp; Lineups</h2>
        {slateMissing ? (
          <p className="rounded-xl border border-amber-900/40 bg-amber-950/20 px-5 py-4 text-sm text-amber-200">
            Slate lineups are missing from the latest export. Run{" "}
            <code>npm run sync-picks</code> to reload all {scheduleGames} games.
          </p>
        ) : null}
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

      {propsAvailable ? (
        <section className="space-y-4">
          <PropPickTable
            picks={topProps}
            title="Top Prop Conviction Plays"
            emptyMessage="No prop conviction plays available for this slate."
          />
        </section>
      ) : (
        <section className="rounded-xl border border-[var(--card-border)] bg-[var(--card)] px-5 py-4 text-sm text-[var(--muted)]">
          Player prop picks are hidden until enough live market data is available.
        </section>
      )}
    </div>
  );
}
