# Sports Picks Blog

Next.js site for publishing daily MLB moneyline and player prop picks from your existing Python projection systems.

## Setup

```powershell
cd C:\Users\Mamas\sports-picks-blog
npm install
copy .env.local.example .env.local
```

Configure paths in `.env.local` if your Python projects live elsewhere.

## GitHub Actions (ISR sync)

Workflow: `.github/workflows/daily-isr-sync.yml`

| Schedule (UTC) | Local (EDT) | Action |
|---|---|---|
| `0 12 * * *` | 8:00 AM | Moneyline + slate sync, commit/push, sleep until T-20 first pitch → ISR revalidate |
| `0 16 * * *` | 12:00 PM | Full sync including props, commit/push |

**Repository secrets required:**

| Secret | Purpose |
|---|---|
| `THE_ODDS_API_KEY` | Live odds + props ingest |
| `REVALIDATION_SECRET` | Protects `/api/revalidate` (set same value in Vercel env) |
| `SITE_URL` | Production URL, e.g. `https://your-app.vercel.app` |

**Vercel env:** add `REVALIDATION_SECRET` matching the GitHub secret.

## Sync picks (local only)

Exports slate and moneyline picks from the internal MLB engine (`engines/mlb_engine/`) into `data/picks/`:

| Command | What it exports | Typical runtime |
|---|---|---|
| `npm run sync-picks` | Slate + live moneylines (no props) | ~5–15 min |
| `npm run sync-picks-full` | Slate + moneylines + live props | ~15–25 min |

```powershell
# Fast daily sync (moneylines + slate only)
npm run sync-picks

# Full export including player props (slow — live API + Statcast)
npm run sync-picks-full
```

Props-only retry (when full export timed out but moneylines already wrote):

```powershell
python scripts/export_daily_picks.py --props-only 2026-07-05 data/picks/.props-temp-2026-07-05.json
```

Then merge into the dated JSON manually or re-run `sync-picks-full`.

Or with the predictor venv directly:

```powershell
& ".\engines\mlb_engine\predictor\.venv\Scripts\python.exe" scripts/export_daily_picks.py --skip-props
```

Optional date override:

```powershell
python scripts/export_daily_picks.py 2026-06-30
python scripts/export_daily_picks.py 2026-06-30 --skip-props
```

**Requirements:** Python venvs under `engines/mlb_engine/predictor` and `engines/mlb_engine/props` with dependencies installed. Live props need API keys in each engine's `.env`. Override props timeout via `PROPS_EXPORT_TIMEOUT_SECONDS` in `.env.local` (default 1800s).

## Run locally

```powershell
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Share with friends (Vercel)

The live site is hosted on Vercel. Pick data comes from JSON files committed to this repo — Vercel does not run the Python sync script.

### First-time deploy

1. Push this repo to GitHub
2. Import the repo at [vercel.com/new](https://vercel.com/new)
3. Deploy with default Next.js settings (no env vars needed)
4. Share your `*.vercel.app` URL

### Daily update workflow

Each morning on your PC:

```powershell
cd C:\Users\Mamas\sports-picks-blog

# 1. Export fresh picks (slate + moneylines; add sync-picks-full for props)
npm run sync-picks

# 2. Or grade yesterday manually after games finish
npm run grade-picks

# 3. Push to GitHub — Vercel auto-rebuilds in ~1–2 min
git add data/picks/ data/results/
git commit -m "Update picks for $(Get-Date -Format yyyy-MM-dd)"
git push
```

Friends see updated picks after the Vercel deployment finishes.

## Pages

| Route | Description |
|---|---|
| `/` | Today's picks — moneyline cards, slate, confidence ratings |
| `/picks/[date]` | Daily archive with full slate |
| `/performance` | Historical backtest metrics |
| `/about` | Methodology and responsible gambling disclaimer |

## Project structure

- `data/picks/` — JSON exports (one file per date + `latest.json`)
- `engines/mlb_engine/` — MLB predictor + props models (internal)
- `engines/wnba_engine/` — WNBA models (placeholder)
- `scripts/export_daily_picks.py` — bridge to Python pick systems
- `lib/` — types and file readers
- `components/` — UI components

## What works where

| Feature | Vercel (public) | Your PC (local) |
|---|---|---|
| View picks site | Yes | Yes |
| `npm run sync-picks` | No | Yes |
| Live props export | No | Yes (when APIs work) |
| Updating picks | Commit + push JSON | Run sync script |
