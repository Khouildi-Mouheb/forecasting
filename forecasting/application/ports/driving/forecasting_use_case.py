from abc import ABC, abstractmethod
from forecasting.domain.forecast_request import ForecastRequest
from forecasting.domain.forecast_result import ForecastResult


class ForecastingUseCase(ABC):
    """Driving Port: Entry point interface for generating forecasts."""

    @abstractmethod
    def generateForecast(self, request: ForecastRequest) -> ForecastResult:
        """Generates a forecast for the given request and historical data."""
        pass
