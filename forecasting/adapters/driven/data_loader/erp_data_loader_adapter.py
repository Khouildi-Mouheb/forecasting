from datetime import datetime, date
from typing import Any, List
import pandas as pd

from forecasting.application.ports.driven.data_loader_port import DataLoaderPort
from forecasting.domain.raw_data import RawData
from forecasting.domain.historical_data_point import HistoricalDataPoint
from forecasting.domain.forecast_request import ForecastRequest


class ERPDataLoaderAdapter(DataLoaderPort):
    """
    Driven Adapter: Implements DataLoaderPort to transform ERP payloads
    (JSON list of records, CSV/DataFrame, or ForecastRequest) into domain RawData.
    """

    def load(self, source: Any) -> RawData:
        """Loads and parses source data into RawData entity."""
        raw_points: List[HistoricalDataPoint] = []

        # If source is a ForecastRequest with history attached
        if isinstance(source, ForecastRequest):
            payload = source.history
        else:
            payload = source

        if payload is None:
            return RawData(points=[])

        # If payload is a pandas DataFrame
        if isinstance(payload, pd.DataFrame):
            for _, row in payload.iterrows():
                dt = self._parse_date(row.get("date") or row.get("ds"))
                qty = float(row.get("quantity") if "quantity" in row else row.get("y", 0.0))
                if dt is not None:
                    raw_points.append(HistoricalDataPoint(date=dt, quantity=qty))
            return RawData(points=raw_points)

        # If payload is a list of dicts or objects
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, HistoricalDataPoint):
                    raw_points.append(item)
                elif isinstance(item, dict):
                    dt_val = item.get("date") or item.get("ds") or item.get("timestamp")
                    qty_val = item.get("quantity") if "quantity" in item else item.get("y", 0.0)
                    dt = self._parse_date(dt_val)
                    if dt is not None:
                        raw_points.append(HistoricalDataPoint(date=dt, quantity=float(qty_val)))

        return RawData(points=raw_points)

    def _parse_date(self, val: Any) -> Any:
        if val is None:
            return None
        if isinstance(val, (date, datetime)):
            return val.date() if isinstance(val, datetime) else val
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00")).date()
            except ValueError:
                return pd.to_datetime(val).date()
        return None

