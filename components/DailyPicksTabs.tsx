"use client";

import type { DailyPicks } from "@/lib/types";

import { PickCard } from "./PickCard";
import { PropPickTable } from "./PropPickTable";

interface DailyPicksTabsProps {
  picks: DailyPicks;
}

type Tab = "moneyline" | "props";

export function DailyPicksTabs({ picks }: DailyPicksTabsProps) {
  const tabs: { id: Tab; label: string; count: number }[] = [
    {
      id: "moneyline",
      label: "Moneyline",
      count: picks.moneylinePicks.length,
    },
    {
      id: "props",
      label: "Player Props",
      count: picks.propPicks.conviction.length,
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-2">
        {tabs.map((tab) => (
          <a
            key={tab.id}
            href={`#${tab.id}`}
            className="rounded-full border border-[var(--card-border)] bg-[var(--card)] px-4 py-2 text-sm text-[var(--muted)] transition hover:border-[var(--accent-muted)] hover:text-white"
          >
            {tab.label} ({tab.count})
          </a>
        ))}
      </div>

      <section id="moneyline" className="scroll-mt-24 space-y-4">
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

      <section id="props" className="scroll-mt-24 space-y-4">
        <PropPickTable picks={picks.propPicks.conviction} />
        {picks.propPicks.batterEdges.length > 0 && (
          <PropPickTable
            picks={picks.propPicks.batterEdges}
            title="All Batter Edges"
          />
        )}
        {picks.propPicks.pitcherEdges.length > 0 && (
          <PropPickTable
            picks={picks.propPicks.pitcherEdges}
            title="All Pitcher Edges"
          />
        )}
      </section>
    </div>
  );
}
