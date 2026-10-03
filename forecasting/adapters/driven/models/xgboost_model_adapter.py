from datetime import datetime, timedelta, timezone
from typing import List, Optional
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from forecasting.application.ports.driven.model_port import ModelPort
from forecasting.domain.prepared_data import PreparedData
from forecasting.domain.forecast_result import ForecastResult


class XGBoostModelAdapter(ModelPort):
    """
    Driven Adapter: Implements ModelPort using Extreme Gradient Boosting (XGBoost).
    Uses recursive multi-step forecasting with engineered lag and calendar features.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.05):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def predict(self, data: PreparedData, horizon: int, item_id: str = "item") -> ForecastResult:
        """Trains XGBoost regressor and produces recursive multi-step forecasts."""
        df = data.features.copy()
        if df.empty or len(df) < 5:
            raise ValueError("Insufficient data to fit XGBoost model.")

        feature_cols = [c for c in df.columns if c not in ["date", "y"]]
        X = df[feature_cols]
        y = df["y"]

        model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y)

        # Recursive multi-step forecasting
        predicted_values: List[float] = []
        current_history = df["y"].tolist()
        last_date = pd.to_datetime(df["date"].iloc[-1])

        lag_cols = [c for c in feature_cols if c.startswith("lag_")]
        rolling_cols = [c for c in feature_cols if c.startswith("rolling_mean_")]

        for step in range(1, horizon + 1):
            next_date = last_date + timedelta(days=step)
            row_dict = {
                "dayofweek": next_date.dayofweek,
                "day": next_date.day,
                "month": next_date.month,
                "is_weekend": 1 if next_date.dayofweek in [5, 6] else 0,
            }

            for col in lag_cols:
                lag_num = int(col.split("_")[1])
                row_dict[col] = current_history[-lag_num] if len(current_history) >= lag_num else current_history[0]

            for col in rolling_cols:
                win = int(col.split("_")[-1])
                row_dict[col] = float(np.mean(current_history[-win:])) if len(current_history) >= win else float(np.mean(current_history))

            rolling_std_cols = [c for c in feature_cols if c.startswith("rolling_std_")]
            for col in rolling_std_cols:
                win = int(col.split("_")[-1])
                row_dict[col] = float(np.std(current_history[-win:])) if len(current_history) >= win else 0.0

            x_next = pd.DataFrame([row_dict])[feature_cols]
            pred = float(model.predict(x_next)[0])
            pred = max(0.0, round(pred, 2))

            predicted_values.append(pred)
            current_history.append(pred)

        return ForecastResult(
            itemId=item_id,
            predictedValues=predicted_values,
            confidenceInterval=None,
            modelUsed="XGBoost",
            generatedAt=datetime.now(timezone.utc),
        )
