from dataclasses import dataclass
from typing import List
from .historical_data_point import HistoricalDataPoint


@dataclass
class RawData:
    points: List[HistoricalDataPoint]
