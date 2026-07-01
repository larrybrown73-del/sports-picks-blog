import fs from "fs";
import path from "path";

import type { DailyPicks, PerformanceSnapshot } from "./types";

const PICKS_DIR = path.join(process.cwd(), "data", "picks");

function readJsonFile<T>(filePath: string): T | null {
  if (!fs.existsSync(filePath)) {
    return null;
  }

  const raw = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(raw) as T;
}

export function getPickDates(): string[] {
  if (!fs.existsSync(PICKS_DIR)) {
    return [];
  }

  return fs
    .readdirSync(PICKS_DIR)
    .filter((file) => file.endsWith(".json") && file !== "latest.json")
    .map((file) => file.replace(".json", ""))
    .sort()
    .reverse();
}

export function getPicksByDate(date: string): DailyPicks | null {
  return readJsonFile<DailyPicks>(path.join(PICKS_DIR, `${date}.json`));
}

export function getLatestPicks(): DailyPicks | null {
  const latest = readJsonFile<DailyPicks>(path.join(PICKS_DIR, "latest.json"));
  if (latest) {
    return latest;
  }

  const dates = getPickDates();
  if (dates.length === 0) {
    return null;
  }

  return getPicksByDate(dates[0]);
}

export function getAllPerformanceSnapshots(): PerformanceSnapshot[] {
  const dates = getPickDates();

  return dates
    .map((date) => {
      const picks = getPicksByDate(date);
      if (!picks) {
        return null;
      }

      return {
        date: picks.date,
        generatedAt: picks.generatedAt,
        performance: picks.performance,
      };
    })
    .filter((snapshot): snapshot is PerformanceSnapshot => snapshot !== null)
    .sort((a, b) => a.date.localeCompare(b.date));
}

export { formatGeneratedAt, formatMarketLabel, parsePercent } from "./utils";
