# WNBA Engine

Placeholder for WNBA projection and prop models.

Add scripts, models, and export helpers here when the WNBA pipeline is ready. Mirror the layout used by `engines/mlb_engine/` (predictor + props subfolders).

## Calibration

- `calibration/brier_score.py` — `WNBABrierCalibrator` for DataFrame post-slate Brier,
  rolling drift, and probability-bin bias reports.
- `calibration/brier_tracker.py` — `WNBABrierTracker` JSON ledger for manual or
  model win probabilities (log → settle → Brier).

The live WNBA pipeline mirrors these under `intel/calibration/` and wires the
calibrator into `intel/audit.py` for nightly `--mode audit` runs.

```bash
# Sandbox (writes test_wnba_ledger.json in CWD)
python -m engines.wnba_engine.calibration.brier_tracker
```
