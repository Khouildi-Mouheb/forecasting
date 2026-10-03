from abc import ABC, abstractmethod
from forecasting.domain.cleaned_data import CleanedData
from forecasting.domain.prepared_data import PreparedData


class DataPreparatorPort(ABC):
    """Driven Port: Interface for transforming cleaned time series data into model-ready features."""

    @abstractmethod
    def prepare(self, data: CleanedData) -> PreparedData:
        """Prepares features from cleaned data."""
        pass
