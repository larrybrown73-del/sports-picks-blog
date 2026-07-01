import type { DailyPicks } from "./types";

export function resolvePickCounts(picks: DailyPicks): {
  slateGames: number;
  valuePicks: number;
  scheduleGames: number | null;
} {
  const meta = picks.meta;
  const slateGames = meta?.slateGames ?? picks.slate?.length ?? 0;
  const valuePicks = meta?.valuePicks ?? picks.moneylinePicks.length;
  const scheduleGames = meta?.scheduleGames ?? null;

  return { slateGames, valuePicks, scheduleGames };
}

export function slateHeaderLabel(picks: DailyPicks): string {
  const { slateGames, valuePicks, scheduleGames } = resolvePickCounts(picks);
  const slateLabel =
    scheduleGames && scheduleGames !== slateGames
      ? `${slateGames} of ${scheduleGames} slate games loaded`
      : `${slateGames}-game slate`;
  return `${slateLabel} · ${valuePicks} value pick${valuePicks === 1 ? "" : "s"}`;
}
