from dataclasses import dataclass
from typing import List
from .historical_data_point import HistoricalDataPoint


@dataclass
class CleanedData:
    points: List[HistoricalDataPoint]
