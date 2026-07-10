import io
import requests
import pandas as pd
import statsapi

url = "https://baseballsavant.mlb.com/leaderboard/outs_above_average?type=Fielder&startYear=2026&endYear=2026&range=year&min=100&csv=true"
r = requests.get(url, timeout=30)
text = r.text.lstrip("\ufeff")
df = pd.read_csv(io.StringIO(text))
print("cols", list(df.columns))
team_col = "display_team_name" if "display_team_name" in df.columns else None
print("team_col", team_col)
if team_col:
    agg = (
        df.groupby(team_col, as_index=False)["outs_above_average"]
        .sum()
        .sort_values("outs_above_average", ascending=False)
    )
    print(agg.head(5))
    print(agg.tail(5))

teams = statsapi.get("teams", {"sportId": 1, "season": 2026}).get("teams", [])
name_map = {t["name"]: t["id"] for t in teams}
abbr_map = {t["abbreviation"]: t["id"] for t in teams}
print("sample maps", list(name_map.items())[:3])
