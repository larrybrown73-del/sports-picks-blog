import Link from "next/link";
import { notFound } from "next/navigation";

import { DailyPicksTabs } from "@/components/DailyPicksTabs";
import { DateSidebar } from "@/components/DateSidebar";
import { PerformanceStatsBar } from "@/components/PerformanceStatsBar";
import { SlateBoard } from "@/components/SlateBoard";
import { formatGeneratedAt, getPicksByDate } from "@/lib/picks";

export const dynamic = "force-dynamic";

interface PickDatePageProps {
  params: Promise<{ date: string }>;
}

export default async function PickDatePage({ params }: PickDatePageProps) {
  const { date } = await params;
  const picks = getPicksByDate(date);

  if (!picks) {
    notFound();
  }

  return (
    <div className="grid gap-8 lg:grid-cols-[240px_1fr]">
      <DateSidebar activeDate={date} />
      <div className="space-y-8">
        <section className="space-y-3">
          <Link href="/" className="text-sm text-[var(--accent)] hover:underline">
            ← Back to today
          </Link>
          <h1 className="text-3xl font-bold text-white">Picks for {date}</h1>
          <p className="text-sm text-[var(--muted)]">
            Exported {formatGeneratedAt(picks.generatedAt)}
          </p>
        </section>

        <PerformanceStatsBar stats={picks.performance} />

        <section className="space-y-4">
          <h2 className="text-xl font-semibold text-white">
            Slate &amp; Lineups ({(picks.slate ?? []).length} games)
          </h2>
          <SlateBoard slate={picks.slate ?? []} />
        </section>

        <DailyPicksTabs picks={picks} />
      </div>
    </div>
  );
}
