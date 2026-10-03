from datetime import date, timedelta
from forecasting.application.services.forecasting_service import ForecastingService
from forecasting.application.services.model_selector import ModelSelector
from forecasting.domain.forecast_request import ForecastRequest
from forecasting.adapters.driven.data_loader.erp_data_loader_adapter import ERPDataLoaderAdapter
from forecasting.adapters.driven.data_cleaner.standard_data_cleaner_adapter import StandardDataCleanerAdapter
from forecasting.adapters.driven.data_preparator.prophet_data_preparator_adapter import ProphetDataPreparatorAdapter
from forecasting.adapters.driven.data_preparator.xgboost_data_preparator_adapter import XGBoostDataPreparatorAdapter
from forecasting.adapters.driven.models.prophet_model_adapter import ProphetModelAdapter
from forecasting.adapters.driven.models.xgboost_model_adapter import XGBoostModelAdapter
from forecasting.adapters.driven.evaluators.mae_evaluator_adapter import MAEEvaluatorAdapter
from forecasting.adapters.driven.evaluators.rmse_evaluator_adapter import RMSEEvaluatorAdapter
from forecasting.adapters.driven.evaluators.mape_evaluator_adapter import MAPEEvaluatorAdapter
from forecasting.adapters.driven.evaluators.r2_evaluator_adapter import R2EvaluatorAdapter


def test_forecasting_service_pipeline():
    base_date = date(2026, 1, 1)
    history = [
        {"date": (base_date + timedelta(days=i)).isoformat(), "quantity": float(10 + (i % 7) * 3)}
        for i in range(30)
    ]

    service = ForecastingService(
        data_loader=ERPDataLoaderAdapter(),
        data_cleaner=StandardDataCleanerAdapter(),
        preparators={
            "Prophet": ProphetDataPreparatorAdapter(),
            "XGBoost": XGBoostDataPreparatorAdapter(),
        },
        models={
            "Prophet": ProphetModelAdapter(),
            "XGBoost": XGBoostModelAdapter(),
        },
        evaluators=[
            MAEEvaluatorAdapter(),
            RMSEEvaluatorAdapter(),
            MAPEEvaluatorAdapter(),
            R2EvaluatorAdapter(),
        ],
        model_selector=ModelSelector(primary_metric="MAE"),
    )

    # Test auto model selection
    req_auto = ForecastRequest(itemId="OLIVE_OIL_1L", horizon=7, history=history)
    result = service.generateForecast(req_auto)
    assert result.itemId == "OLIVE_OIL_1L"
    assert len(result.predictedValues) == 7
    assert result.modelUsed in ["Prophet", "XGBoost"]

    # Test explicit model choice: XGBoost
    req_xgb = ForecastRequest(itemId="OLIVE_OIL_1L", horizon=5, modelType="XGBoost", history=history)
    result_xgb = service.generateForecast(req_xgb)
    assert result_xgb.modelUsed == "XGBoost"
    assert len(result_xgb.predictedValues) == 5
