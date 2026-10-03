from abc import ABC, abstractmethod
from typing import Any
from forecasting.domain.raw_data import RawData


class DataLoaderPort(ABC):
    """Driven Port: Interface for loading/ingesting raw data from a source."""

    @abstractmethod
    def load(self, source: Any) -> RawData:
        """Loads and converts external source payload into domain RawData."""
        pass
