import fs from "fs";
import path from "path";

import type { WnbaDailyPicks } from "./types";

const WNBA_PICKS_DIR = path.join(process.cwd(), "data", "picks", "wnba");

function readJsonFile<T>(filePath: string): T | null {
  if (!fs.existsSync(filePath)) {
    return null;
  }
  const raw = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(raw) as T;
}

export function getWnbaPickDates(): string[] {
  if (!fs.existsSync(WNBA_PICKS_DIR)) {
    return [];
  }

  return fs
    .readdirSync(WNBA_PICKS_DIR)
    .filter((file) => file.endsWith(".json") && file !== "latest.json")
    .map((file) => file.replace(".json", ""))
    .sort()
    .reverse();
}

export function getWnbaPicksByDate(date: string): WnbaDailyPicks | null {
  return readJsonFile<WnbaDailyPicks>(path.join(WNBA_PICKS_DIR, `${date}.json`));
}

export function getLatestWnbaPicks(): WnbaDailyPicks | null {
  const latest = readJsonFile<WnbaDailyPicks>(path.join(WNBA_PICKS_DIR, "latest.json"));
  if (latest) {
    return latest;
  }

  const dates = getWnbaPickDates();
  if (dates.length === 0) {
    return null;
  }

  return getWnbaPicksByDate(dates[0]);
}

export function formatWnbaTip(iso: string): string {
  if (!iso) {
    return "TBD";
  }
  const date = new Date(iso);
  return date.toLocaleString("en-US", {
    timeZone: "America/New_York",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
}
