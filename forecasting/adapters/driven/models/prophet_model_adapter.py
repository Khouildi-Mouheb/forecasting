from datetime import datetime, timezone
from typing import Optional
import logging
import pandas as pd
from prophet import Prophet

from forecasting.application.ports.driven.model_port import ModelPort
from forecasting.domain.prepared_data import PreparedData
from forecasting.domain.forecast_result import ForecastResult

# Suppress cmdstanpy / prophet verbose logs
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


class ProphetModelAdapter(ModelPort):
    """
    Driven Adapter: Implements ModelPort using Meta Prophet.
    Produces forecasts with Bayesian uncertainty intervals.
    """

    def __init__(self, weekly_seasonality: bool = True, yearly_seasonality: bool = False):
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality

    def predict(self, data: PreparedData, horizon: int, item_id: str = "item") -> ForecastResult:
        """Fits Prophet and produces forecasts along with confidence intervals."""
        df = data.features
        if df.empty or len(df) < 2:
            raise ValueError("Insufficient data to fit Prophet model.")

        # Ensure correct column naming and datetime typing
        m_df = pd.DataFrame()
        m_df["ds"] = pd.to_datetime(df["ds"])
        m_df["y"] = df["y"].astype(float)

        # Fit model
        model = Prophet(
            weekly_seasonality=self.weekly_seasonality and len(m_df) >= 14,
            yearly_seasonality=self.yearly_seasonality and len(m_df) >= 365,
            daily_seasonality=False,
            interval_width=0.95,
        )
        model.fit(m_df)

        # Create future dataframe and predict
        future = model.make_future_dataframe(periods=horizon, freq="D", include_history=False)
        forecast = model.predict(future)

        predicted_values = [max(0.0, round(float(v), 2)) for v in forecast["yhat"].tolist()]
        lower_bounds = [max(0.0, round(float(v), 2)) for v in forecast["yhat_lower"].tolist()]
        upper_bounds = [max(0.0, round(float(v), 2)) for v in forecast["yhat_upper"].tolist()]

        return ForecastResult(
            itemId=item_id,
            predictedValues=predicted_values,
            confidenceInterval={
                "lower": lower_bounds,
                "upper": upper_bounds,
            },
            modelUsed="Prophet",
            generatedAt=datetime.now(timezone.utc),
        )
