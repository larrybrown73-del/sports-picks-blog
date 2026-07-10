from pybaseball import fielding_stats, team_fielding

print("team_fielding cols sample")
try:
    df = team_fielding(2025, 2025)
    print(df.columns.tolist()[:30])
    print(df.head(3))
except Exception as exc:
    print("team_fielding failed", exc)

print("fielding_stats cols sample")
try:
    df2 = fielding_stats(2025, 2025, qual=100)
    print(df2.columns.tolist())
    oaa_cols = [c for c in df2.columns if "OAA" in str(c).upper() or "DRS" in str(c).upper()]
    print("oaa/drs cols", oaa_cols)
except Exception as exc:
    print("fielding_stats failed", exc)
