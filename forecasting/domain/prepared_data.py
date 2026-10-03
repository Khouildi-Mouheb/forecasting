from dataclasses import dataclass
import pandas as pd


@dataclass
class PreparedData:
    features: pd.DataFrame
