import datetime as _dt
from typing import List, Optional
from pydantic import BaseModel, Field


class HistoricalPointSchema(BaseModel):
    date: _dt.date = Field(..., description="Date of the observation (YYYY-MM-DD)")
    quantity: float = Field(..., description="Inventory consumption / demand quantity")


class ForecastRequestSchema(BaseModel):
    itemId: str = Field(..., description="Unique item / SKU identifier")
    horizon: int = Field(..., ge=1, le=365, description="Number of future days to forecast")
    modelType: Optional[str] = Field(None, description="Optional forced model: 'Prophet' or 'XGBoost'")
    history: List[HistoricalPointSchema] = Field(..., min_length=2, description="List of historical data points")


class ConfidenceIntervalSchema(BaseModel):
    lower: List[float] = Field(default_factory=list, description="Lower prediction bounds")
    upper: List[float] = Field(default_factory=list, description="Upper prediction bounds")


class ForecastResponseSchema(BaseModel):
    itemId: str
    predictedValues: List[float]
    confidenceInterval: Optional[ConfidenceIntervalSchema] = None
    modelUsed: str
    generatedAt: _dt.datetime
