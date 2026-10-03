from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class HistoricalDataPoint:
    date: date
    quantity: float
