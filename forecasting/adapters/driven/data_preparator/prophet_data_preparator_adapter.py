import pandas as pd

from forecasting.application.ports.driven.data_preparator_port import DataPreparatorPort
from forecasting.domain.cleaned_data import CleanedData
from forecasting.domain.prepared_data import PreparedData


class ProphetDataPreparatorAdapter(DataPreparatorPort):
    """
    Driven Adapter: Implements DataPreparatorPort to structure data into
    Prophet-compatible format with 'ds' and 'y' columns.
    """

    def prepare(self, data: CleanedData) -> PreparedData:
        """Transforms cleaned points into a DataFrame with 'ds' and 'y' columns."""
        if not data.points:
            return PreparedData(features=pd.DataFrame(columns=["ds", "y"]))

        df = pd.DataFrame([
            {"ds": pd.to_datetime(p.date), "y": float(p.quantity)}
            for p in data.points
        ])
        df = df.sort_values(by="ds").reset_index(drop=True)
        return PreparedData(features=df)
