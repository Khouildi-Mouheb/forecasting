from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict


@dataclass
class ForecastResult:
    itemId: str
    predictedValues: List[float]
    confidenceInterval: Optional[Dict[str, List[float]]] = None
    modelUsed: str = ""
    generatedAt: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
