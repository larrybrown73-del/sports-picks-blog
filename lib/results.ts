import fs from "fs";
import path from "path";

import type { DailyResults } from "./types";

const RESULTS_DIR = path.join(process.cwd(), "data", "results");

function readJsonFile<T>(filePath: string): T | null {
  if (!fs.existsSync(filePath)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(filePath, "utf-8")) as T;
}

export function getResultDates(): string[] {
  if (!fs.existsSync(RESULTS_DIR)) {
    return [];
  }

  return fs
    .readdirSync(RESULTS_DIR)
    .filter((file) => file.endsWith(".json"))
    .map((file) => file.replace(".json", ""))
    .sort()
    .reverse();
}

export function getResultsByDate(date: string): DailyResults | null {
  return readJsonFile<DailyResults>(path.join(RESULTS_DIR, `${date}.json`));
}

export function getLatestResults(): DailyResults | null {
  const dates = getResultDates();
  if (dates.length === 0) {
    return null;
  }
  return getResultsByDate(dates[0]);
}

export function getYesterdayResults(todayIso?: string): DailyResults | null {
  const today = todayIso ? new Date(`${todayIso}T12:00:00`) : new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const iso = yesterday.toISOString().slice(0, 10);

  const exact = getResultsByDate(iso);
  if (exact) {
    return exact;
  }

  return getLatestResults();
}
