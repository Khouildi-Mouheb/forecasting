from abc import ABC, abstractmethod
from forecasting.domain.prepared_data import PreparedData
from forecasting.domain.forecast_result import ForecastResult


class ModelPort(ABC):
    """Driven Port: Interface for forecasting model adapters (Prophet, XGBoost)."""

    @abstractmethod
    def predict(self, data: PreparedData, horizon: int) -> ForecastResult:
        """Generates predictions for a specified horizon from prepared data."""
        pass
