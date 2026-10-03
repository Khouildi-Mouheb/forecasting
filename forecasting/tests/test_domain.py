from datetime import date
import pandas as pd
from forecasting.domain.historical_data_point import HistoricalDataPoint
from forecasting.domain.raw_data import RawData
from forecasting.domain.cleaned_data import CleanedData
from forecasting.domain.prepared_data import PreparedData
from forecasting.domain.forecast_request import ForecastRequest
from forecasting.domain.forecast_result import ForecastResult
from forecasting.domain.evaluation_result import EvaluationResult


def test_domain_entities_creation():
    dp = HistoricalDataPoint(date=date(2026, 1, 1), quantity=15.0)
    assert dp.date == date(2026, 1, 1)
    assert dp.quantity == 15.0

    raw = RawData(points=[dp])
    assert len(raw.points) == 1

    cleaned = CleanedData(points=[dp])
    assert len(cleaned.points) == 1

    df = pd.DataFrame({"y": [15.0]})
    prep = PreparedData(features=df)
    assert len(prep.features) == 1

    req = ForecastRequest(itemId="OIL_001", horizon=7, modelType="Prophet")
    assert req.itemId == "OIL_001"
    assert req.horizon == 7
    assert req.modelType == "Prophet"

    res = ForecastResult(
        itemId="OIL_001",
        predictedValues=[12.0, 14.5],
        confidenceInterval={"lower": [10.0, 12.0], "upper": [14.0, 16.0]},
        modelUsed="Prophet",
    )
    assert res.itemId == "OIL_001"
    assert len(res.predictedValues) == 2

    ev = EvaluationResult(metricName="MAE", value=1.25)
    assert ev.metricName == "MAE"
    assert ev.value == 1.25
