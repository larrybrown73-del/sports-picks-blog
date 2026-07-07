# Baseball Player Props Projection Model

Base Rate + Matchup Adjustment model for projecting batter rate stats and opportunity on a daily slate.

## Setup

```powershell
cd "d:\Juniors Files\baseball-props-model"
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set API keys (`.env.example` is a template only—never commit real keys):

- `RUNDOWN_API_KEY` — primary source for Vegas totals and slate event IDs (TheRundown)
- `THE_ODDS_API_KEY` — player props fallback when TheRundown tier omits prop markets (required for live slate)
- `SERP_API_KEY` — optional; used when The Odds API returns 401 (quota exhausted)

If an API key is ever exposed in chat or logs, rotate it in the provider dashboard and update `.env`.

```powershell
copy .env.example .env
```

## Run slate (unified pipeline)

One command runs the full stack: injuries + rust ramps, batter/pitcher projections, live markets, and conviction edges.

**Live slate** (default — MLB lineups, pybaseball/statcast, TheRundown + market fallbacks):

```powershell
py scripts/run_mock_slate.py
py scripts/run_mock_slate.py --date 2025-06-27
```

**Mock slate** (offline, no API keys — pass explicitly for tests or local dev):

```powershell
py scripts/run_mock_slate.py --source mock
```

Add `--verbose` to surface INFO-level API diagnostics (TheRundown cache hits, props fallback counts). Default output shows edge sheets and warnings only.

Add `--export-csv PATH` to write `*_batter_edges.csv`, `*_pitcher_edges.csv`, and `*_conviction.csv` (directory or file prefix).

Output sections:

1. **Batter Total Bases Projections & Market Edges** — Proj TB vs market line, Over/Under odds, model probability, Edge %, and recommendation.
2. **Pitcher Outs & Workload Projections** — capped outs (27.0 max), pitch count, 5.3 pitches/out baseline, market line, Edge %, and recommendation.
3. **Model Highest Conviction Predictions** — top plays from both sheets ranked by absolute Edge %.

Live mode resolves all slate games to market event IDs via TheRundown, fetches props through TheRundown → Odds API → stale cache, and ranks conviction by Edge % across `batter_total_bases` and `pitcher_outs`.

### Fetch slate + canvas export

Focused live workflow: run the full model, export `canvas_games.json` / `canvas_betting_intel.json`, and sync the canvas TSX. Defaults to TB Over 1.5, top 10 conviction plays, and skips pitch-location fetches for speed (~4–10 min on a full slate).

```powershell
py scripts/fetch_slate.py
py scripts/fetch_slate.py --date 2026-07-03
py scripts/fetch_slate.py --no-sync-canvas
```

Requires `THE_ODDS_API_KEY` in `.env` (and optionally `RUNDOWN_API_KEY` for primary props/Vegas feeds).

Programmatic access: `from baseball_props.pipeline import run_slate`. `run_slate()` and `load_slate_frames()` default to live; use `source="mock"` for offline runs. Tests that need mock data must pass `run_slate(source="mock")` explicitly.

## Run tests

```powershell
py -m pytest tests/ -v
```

## Architecture

| Module | Purpose |
|---|---|
| `baseball_props/pipeline/slate_run.py` | Unified end-to-end slate orchestrator |
| `baseball_props/core/adjustments.py` | Odds-ratio rate projection + log-odds probability helpers |
| `baseball_props/core/baselines.py` | Weighted 14/30/season baseline builder with fallbacks |
| `baseball_props/data/schemas.py` | DataFrame column contracts and validators |
| `baseball_props/data/mock_slate.py` | Single 2-game mock slate |
| `baseball_props/data/ingest.py` | Slate ingest, Vegas totals, team/event ID matching |
| `baseball_props/data/therundown.py` | TheRundown API adapters (Vegas, props, events) |
| `baseball_props/matchups/splits.py` | Handedness split resolution |
| `baseball_props/environment/factors.py` | Park + weather multipliers |
| `baseball_props/opportunity/batters.py` | PA projection from lineup + Vegas totals |
| `baseball_props/analysis/edge_sheets.py` | Batter/pitcher edge sheets and conviction aggregator |
| `baseball_props/analysis/pitcher_projection.py` | Pitcher outs / pitch-count projection with guardrails |

## Core formula

Projected rate (odds ratio):

```
(player_rate * opponent_rate_allowed) / league_avg
```

For bounded stats (K%, BB%), the same relationship is applied in log-odds space via `calculate_projected_probability`.