from typing import List
from datetime import timedelta
import pandas as pd

from forecasting.application.ports.driven.data_cleaner_port import DataCleanerPort
from forecasting.domain.raw_data import RawData
from forecasting.domain.cleaned_data import CleanedData
from forecasting.domain.historical_data_point import HistoricalDataPoint


class StandardDataCleanerAdapter(DataCleanerPort):
    """
    Driven Adapter: Implements DataCleanerPort to handle missing values,
    duplicate dates, irregular intervals, and negative quantities.
    """

    def clean(self, data: RawData) -> CleanedData:
        """Cleans and regularizes raw historical time series data."""
        if not data.points:
            return CleanedData(points=[])

        # Convert to pandas DataFrame for robust cleaning
        records = [{"date": p.date, "quantity": p.quantity} for p in data.points]
        df = pd.DataFrame(records)

        # Ensure date format and sort chronologically
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by="date").reset_index(drop=True)

        # Aggregate duplicates occurring on the same date (sum of demands/transactions)
        df = df.groupby("date", as_index=False)["quantity"].sum()

        # Handle negative values (replace with 0 for inventory/demand consistency)
        df["quantity"] = df["quantity"].clip(lower=0.0)

        # Regularize to a daily frequency grid and fill missing dates with 0.0
        df = df.set_index("date")
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
        df = df.reindex(full_idx, fill_value=0.0)
        df.index.name = "date"
        df = df.reset_index()

        cleaned_points: List[HistoricalDataPoint] = [
            HistoricalDataPoint(date=row["date"].date(), quantity=float(row["quantity"]))
            for _, row in df.iterrows()
        ]

        return CleanedData(points=cleaned_points)
