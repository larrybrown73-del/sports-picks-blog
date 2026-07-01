export interface MoneylinePick {
  awayTeam: string;
  homeTeam: string;
  play: string;
  edgePct: number;
  sizingPct: number;
  modelWinProb?: number;
  confidenceScore?: number;
  confidenceLabel?: "Low" | "Medium" | "High" | "Elite";
}

export interface PropPick {
  player: string;
  market: string;
  line: number;
  recommendation: string;
  edgePct: number;
  modelProb: number | null;
  modelValue?: number | null;
}

export interface PropPicks {
  conviction: PropPick[];
  batterEdges: PropPick[];
  pitcherEdges: PropPick[];
}

export interface LineupPlayer {
  slot: number;
  name: string;
  playerId: string;
}

export interface SlateGame {
  gameId: string;
  awayTeam: string;
  homeTeam: string;
  awayAbbrev: string;
  homeAbbrev: string;
  awayPitcher: string | null;
  homePitcher: string | null;
  awayLineup: LineupPlayer[];
  homeLineup: LineupPlayer[];
}

export interface PerformanceStats {
  accuracy: string;
  roi: string;
  netProfit: string;
  brierScore: string;
  gamesScored: number;
  betsPlaced: number;
}

export interface DailyPicks {
  date: string;
  generatedAt: string;
  slate: SlateGame[];
  moneylinePicks: MoneylinePick[];
  propPicks: PropPicks;
  performance: PerformanceStats;
}

export interface PerformanceSnapshot {
  date: string;
  generatedAt: string;
  performance: PerformanceStats;
}
