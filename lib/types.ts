export interface MoneylinePick {
  matchup: string;
  homeTeam: string;
  awayTeam: string;
  pick: string;
  sportsbook: string;
  betType: string;
  league: string;
  homeAway: "Home" | "Away" | string;
  confidence: number;
  confidenceLabel?: "Low" | "Medium" | "High" | "Elite" | string;
  edge: number;
  closingLineValue: number;
  expectedValue: number;
  winProbability: number;
  edgeDecimal?: number;
  openingLine: number;
  closingLine: number;
  impliedProbability: number;
  result: "Win" | "Loss" | "Pending" | string;
  profitLoss: number;
  unitsWon: number;
  weather: string;
  temperature: number;
  umpire: string;
  startingPitcher: string;
  opposingPitcher: string;
  /** @deprecated legacy export fields — kept optional for older dated JSON */
  play?: string;
  book?: string;
  edgePct?: number;
  sizingPct?: number;
  americanOdds?: number;
  modelWinProb?: number;
  confidenceLabel?: "Low" | "Medium" | "High" | "Elite";
  confidenceScore?: number;
  confidenceTier?: string;
}

export interface PropPick {
  player: string;
  market: string;
  line: number;
  recommendation: string;
  edgePct: number;
  modelProb: number | null;
  modelValue?: number | null;
  evPerUnit?: number | null;
  fractionalKellyPct?: number | null;
  confidenceTier?: string | null;
  confidenceScore?: number | null;
  dataWarnings?: string[];
  verdict?: string | null;
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

export interface PickMeta {
  slateGames: number;
  valuePicks: number;
  propsAvailable: boolean;
  scheduleGames?: number;
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
  meta?: PickMeta;
  slate?: SlateGame[];
  moneylinePicks: MoneylinePick[];
  propPicks: PropPicks;
  performance: PerformanceStats;
}

export interface PerformanceSnapshot {
  date: string;
  generatedAt: string;
  performance: PerformanceStats;
}

export type GradedPickResult = "Win" | "Loss" | "Push" | "Pending" | "NoLine";

export interface GradedPick {
  date: string;
  matchup: string;
  pick: string;
  americanOdds: number;
  result: GradedPickResult;
  profitLoss: number;
  stake: number;
}

export interface DailyResults {
  date: string;
  generatedAt: string;
  unitStake: number;
  picks: GradedPick[];
  summary: {
    wins: number;
    losses: number;
    pending: number;
    netProfitLoss: number;
  };
}
