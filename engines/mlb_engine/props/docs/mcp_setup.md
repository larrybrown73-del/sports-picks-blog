# MCP Workspace Setup — Baseball Props Model

This document describes how MCP servers and visualization schemas align with the **hits prop** prediction pipeline.

## Memory server (`memory-context`)

Configured in [`.cursor/mcp.json`](../.cursor/mcp.json):

> **memory-context** — Persistent memory bank to lock in our model's guardrails, hits prop rules, and pipeline constraints.

Use this server to persist:

- Guardrail thresholds from `baseball_props/config.py` (`HITS_PROP_*`, contact filter floors)
- Pipeline constraints (supported markets, line clamps, conviction filters)
- Agent notes from slate runs and backtests

## Primary prop validation matrix (hits)

The model **no longer optimizes Total Bases props**. All batter edge sheets, conviction ranking, parlay legs, and canvas exports target **hit probability thresholds**:

| Market line | Guardrail path | Sigma | Contact filter |
|-------------|----------------|-------|----------------|
| **Over 0.5 Hits** | `evaluate_hits_prop` | `EDGE_HITS_SIGMA` (0.42) | 15-game K% &lt; 18% + Contact% or BABIP bonus |
| **Over 1.5 Hits** | `evaluate_hits_prop` | `EDGE_HITS_SIGMA` (0.42) | Same contact reward (no singles penalty) |

Implementation entry points:

- `baseball_props/analysis/guardrails.py` — `evaluate_hits_prop(player_id, opponent_pitcher_id, game_context)`
- `baseball_props/analysis/edge_row_builder.py` — row builder; clamps to `HITS_PROP_TARGET_LINES`
- `baseball_props/analysis/edge_sheets.py` — `build_batter_hits_edge_sheet`; quotes from `batter_hits` aggregator payload

Non-clamped lines use continuous normal edge math only (no full checklist).

## Visualization schemas

Canvas and JSON exports (`canvas_games.json`, `canvas_betting_intel.json`) treat **`proj_hits`** as the primary model value for batter rows. Conviction plays use market `batter_hits` with lines **0.5** or **1.5**.

TypeScript canvas types live in the project canvas file under `.cursor/projects/.../canvases/` — `GameBatterRow.proj_hits` is required; `proj_tb` is optional legacy.

## Running a compilation / test check

```bash
cd "D:\Juniors Files\baseball-props-model"
python -m compileall baseball_props/analysis/guardrails.py baseball_props/analysis/edge_row_builder.py
pytest tests/test_edge_row_builder.py tests/test_hits_prop_filter.py -q
```
