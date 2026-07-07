# Baseball Predictor

Streamlit app that trains a Random Forest model on real MLB game data and predicts home/away win probabilities.

## Setup

```powershell
cd "d:\Juniors Files\baseball-predictor"
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## How it works

1. Fetches completed regular-season games from the [MLB Stats API](https://statsapi.mlb.com/) via [MLB-StatsAPI](https://github.com/toddrob99/MLB-StatsAPI).
2. Builds rolling averages for runs scored/allowed (last N games) plus season OBP as features.
3. Adds **temperature** and **wind speed** at each home ballpark from the [Open-Meteo](https://open-meteo.com/) API (cached locally in `cache/weather_cache.json`).
4. Trains two Random Forest regressors to predict home and away runs, then shows derived winner metrics.
5. Lets you pick any home/away matchup or predict today's scheduled games.

## Limitations

- Early-season games are excluded when a team has fewer than 5 prior games in the rolling window.
- OBP uses full-season team stats as a proxy, not true pre-game cumulative OBP.
- Real baseball is noisy; expect roughly 52–58% test accuracy.
- Requires internet access to reach the MLB Stats API.
