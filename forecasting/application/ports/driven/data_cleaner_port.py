from abc import ABC, abstractmethod
from forecasting.domain.raw_data import RawData
from forecasting.domain.cleaned_data import CleanedData


class DataCleanerPort(ABC):
    """Driven Port: Interface for cleaning and validating raw time series data."""

    @abstractmethod
    def clean(self, data: RawData) -> CleanedData:
        """Cleans raw data points, handling missing values, duplicates, and outliers."""
        pass
