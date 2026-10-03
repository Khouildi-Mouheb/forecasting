from datetime import date, timedelta
import pytest
from forecasting.domain.raw_data import RawData
from forecasting.domain.cleaned_data import CleanedData
from forecasting.domain.historical_data_point import HistoricalDataPoint
from forecasting.adapters.driven.data_loader.erp_data_loader_adapter import ERPDataLoaderAdapter
from forecasting.adapters.driven.data_cleaner.standard_data_cleaner_adapter import StandardDataCleanerAdapter
from forecasting.adapters.driven.data_preparator.prophet_data_preparator_adapter import ProphetDataPreparatorAdapter
from forecasting.adapters.driven.data_preparator.xgboost_data_preparator_adapter import XGBoostDataPreparatorAdapter
from forecasting.adapters.driven.evaluators.mae_evaluator_adapter import MAEEvaluatorAdapter
from forecasting.adapters.driven.evaluators.rmse_evaluator_adapter import RMSEEvaluatorAdapter
from forecasting.adapters.driven.evaluators.mape_evaluator_adapter import MAPEEvaluatorAdapter
from forecasting.adapters.driven.evaluators.r2_evaluator_adapter import R2EvaluatorAdapter


def test_data_loader_and_cleaner():
    loader = ERPDataLoaderAdapter()
    payload = [
        {"date": "2026-01-01", "quantity": 10.0},
        {"date": "2026-01-01", "quantity": 5.0}, # Duplicate date
        {"date": "2026-01-03", "quantity": 20.0}, # Gaps between 1 and 3
    ]
    raw = loader.load(payload)
    assert len(raw.points) == 3

    cleaner = StandardDataCleanerAdapter()
    cleaned = cleaner.clean(raw)
    
    # After cleaning: dates should be 2026-01-01 (15.0), 2026-01-02 (0.0 fill), 2026-01-03 (20.0)
    assert len(cleaned.points) == 3
    assert cleaned.points[0].quantity == 15.0
    assert cleaned.points[1].quantity == 0.0
    assert cleaned.points[2].quantity == 20.0


def test_preparators():
    base_date = date(2026, 1, 1)
    points = [HistoricalDataPoint(date=base_date + timedelta(days=i), quantity=float(i * 2 + 5)) for i in range(15)]
    cleaned = CleanedData(points=points)

    prophet_prep = ProphetDataPreparatorAdapter()
    p_data = prophet_prep.prepare(cleaned)
    assert "ds" in p_data.features.columns
    assert "y" in p_data.features.columns
    assert len(p_data.features) == 15

    xgb_prep = XGBoostDataPreparatorAdapter()
    x_data = xgb_prep.prepare(cleaned)
    assert "y" in x_data.features.columns
    assert "dayofweek" in x_data.features.columns
    assert "lag_1" in x_data.features.columns
    assert len(x_data.features) == 15


def test_evaluators():
    actual = [10.0, 20.0, 30.0, 40.0]
    pred = [11.0, 19.0, 31.0, 39.0]

    mae = MAEEvaluatorAdapter().evaluate(actual, pred)
    assert mae.metricName == "MAE"
    assert mae.value == 1.0

    rmse = RMSEEvaluatorAdapter().evaluate(actual, pred)
    assert rmse.metricName == "RMSE"
    assert round(rmse.value, 2) == 1.0

    mape = MAPEEvaluatorAdapter().evaluate(actual, pred)
    assert mape.metricName == "MAPE"
    assert mape.value > 0

    r2 = R2EvaluatorAdapter().evaluate(actual, pred)
    assert r2.metricName == "R2"
    assert r2.value > 0.9
