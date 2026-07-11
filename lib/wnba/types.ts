export interface WnbaPlayerProjection {
  name: string;
  pos: string;
  pts: number;
  reb: number;
  ast: number;
  pra: number;
}

export interface WnbaSpreadPick {
  team: string;
  spread: number;
  odds: number;
  margin: number;
  coverProb: number;
  ev: number;
  edge: number;
  kelly: number;
}

export interface WnbaPropLean {
  player: string;
  team: string;
  stat: string;
  direction: string;
  strength: string;
  line: number;
  projection: number;
  delta: number;
  prob: number;
  odds: number;
  ev: number;
  edge: number;
  kelly: number;
}

export interface WnbaApprovedBet {
  market: string;
  team: string;
  player: string | null;
  stat: string | null;
  direction: string;
  line: number | null;
  odds: number;
  prob: number;
  ev: number;
  edge: number;
  kelly: number;
}

export interface WnbaGame {
  id: string;
  date: string;
  away: string;
  home: string;
  tip: string;
  status: string;
  teams: Record<string, WnbaPlayerProjection[]>;
  spreads: WnbaSpreadPick[];
  props: WnbaPropLean[];
  approved: WnbaApprovedBet[];
}

export interface WnbaDailyPicks {
  date: string;
  generated_at: string;
  generatedAt: string;
  models_trained_on: number;
  run_id: string;
  phase: string;
  no_bet: boolean;
  no_bet_reason: string;
  spreads_error: string | null;
  props_error: string | null;
  approved_count: number;
  rejected_count: number;
  approved_bets: WnbaApprovedBet[];
  games: WnbaGame[];
}
