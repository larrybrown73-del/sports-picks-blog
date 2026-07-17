"""Brier Score calibration for WNBA post-slate audits."""

import pandas as pd
from sklearn.metrics import brier_score_loss


class WNBABrierCalibrator:
    """
    Handles post-slate audit tracking by calculating historical
    Brier Score calibration and adjusting future probability mappings.
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size

    def calculate_slate_brier(self, df_slate: pd.DataFrame) -> float:
        """
        Computes the flat Brier Score for a single night's slate.

        Expects columns: ``predicted_prob`` (0.0-1.0) and ``actual_outcome`` (1 or 0).
        """
        if df_slate.empty:
            return 0.0

        y_true = df_slate["actual_outcome"].to_numpy()
        y_prob = df_slate["predicted_prob"].to_numpy()
        return float(brier_score_loss(y_true, y_prob))

    def compute_rolling_calibration(self, historical_log_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates a moving Brier Score to audit whether the model is getting
        sharper or drifting into uncalibrated noise over time.

        Expects columns: ``game_date`` and ``brier_error_delta`` ((prob - actual)^2).
        """
        df = historical_log_df.sort_values(by="game_date").copy()
        df["rolling_brier"] = (
            df["brier_error_delta"]
            .rolling(window=self.window_size, min_periods=10)
            .mean()
        )
        return df

    def get_calibration_bias(self, df_history: pd.DataFrame, bins: int = 5) -> dict:
        """
        Breaks predictions into percentage bins to spot structural bias.
        Shows whether an 80% favorite is actually winning ~80% of the time.
        """
        if df_history.empty or "predicted_prob" not in df_history.columns:
            return {}

        df = df_history.copy()
        n_unique = df["predicted_prob"].nunique(dropna=True)
        if n_unique < 2:
            return {}

        q = min(bins, n_unique)
        df["bin"] = pd.qcut(df["predicted_prob"], q=q, labels=False, duplicates="drop")

        bias_report: dict = {}
        for b in sorted(df["bin"].dropna().unique()):
            bin_data = df[df["bin"] == b]
            avg_pred = bin_data["predicted_prob"].mean()
            actual_win_rate = bin_data["actual_outcome"].mean()
            bias_report[f"Bin_{int(b)}"] = {
                "avg_projected_prob": round(float(avg_pred), 3),
                "actual_realized_rate": round(float(actual_win_rate), 3),
                "calibration_gap": round(float(avg_pred - actual_win_rate), 3),
                "n": int(len(bin_data)),
            }

        return bias_report

    @staticmethod
    def per_row_brier_delta(predicted_prob: pd.Series, actual_outcome: pd.Series) -> pd.Series:
        """Per-observation squared error for rolling calibration logs."""
        return (predicted_prob.astype(float) - actual_outcome.astype(float)) ** 2
