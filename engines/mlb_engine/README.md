# MLB Engine

Internal hub for MLB predictive models used by `scripts/export_daily_picks.py`.

| Directory | Source | Role |
|-----------|--------|------|
| `predictor/` | baseball-predictor | Moneylines, slate evaluation, backtest |
| `props/` | baseball-props-model | Player hits props, conviction, parlay builder |

## Setup

```powershell
cd engines/mlb_engine/predictor
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

cd ..\props
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

Copy API keys into each `.env` (not committed):

- `predictor/.env` — `THE_ODDS_API_KEY`, etc.
- `props/.env` — `THE_ODDS_API_KEY`, etc.

## Paths

The blog export script defaults to these folders. Override in `.env.local` if needed:

```
BASEBALL_PREDICTOR_PATH=engines/mlb_engine/predictor
BASEBALL_PROPS_PATH=engines/mlb_engine/props
```
