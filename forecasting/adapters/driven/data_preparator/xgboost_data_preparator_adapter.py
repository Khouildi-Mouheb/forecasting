import pandas as pd
import numpy as np

from forecasting.application.ports.driven.data_preparator_port import DataPreparatorPort
from forecasting.domain.cleaned_data import CleanedData
from forecasting.domain.prepared_data import PreparedData


class XGBoostDataPreparatorAdapter(DataPreparatorPort):
    """
    Driven Adapter: Implements DataPreparatorPort to construct tabular features
    (lags, rolling statistics, calendar indicators) for gradient-boosted models.
    """

    def prepare(self, data: CleanedData) -> PreparedData:
        """Transforms cleaned points into a tabular feature set."""
        features_df = self.buildFeatures(data)
        return PreparedData(features=features_df)

    def buildFeatures(self, data: CleanedData) -> pd.DataFrame:
        """
        Builds lag, rolling, and temporal calendar features from cleaned time series.
        """
        if not data.points:
            return pd.DataFrame()

        df = pd.DataFrame([
            {"date": pd.to_datetime(p.date), "y": float(p.quantity)}
            for p in data.points
        ])
        df = df.sort_values(by="date").reset_index(drop=True)

        # Calendar features
        df["dayofweek"] = df["date"].dt.dayofweek
        df["day"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        # Lag features
        n_obs = len(df)
        max_lags = min(7, max(1, n_obs // 4))
        for lag in range(1, max_lags + 1):
            df[f"lag_{lag}"] = df["y"].shift(lag)

        # Rolling statistics
        rolling_win = min(7, max(2, n_obs // 4))
        df[f"rolling_mean_{rolling_win}"] = df["y"].shift(1).rolling(window=rolling_win, min_periods=1).mean()
        df[f"rolling_std_{rolling_win}"] = df["y"].shift(1).rolling(window=rolling_win, min_periods=1).std().fillna(0.0)

        # Forward fill and backward fill any edge NaNs produced by shifting
        df = df.bfill().ffill()

        return df
