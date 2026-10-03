from dataclasses import dataclass
from typing import Optional, Any


@dataclass(frozen=True)
class ForecastRequest:
    itemId: str
    horizon: int
    modelType: Optional[str] = None
    history: Optional[Any] = None
