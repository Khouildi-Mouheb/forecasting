from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Domain & Ports
from forecasting.config.settings import settings
from forecasting.application.services.forecasting_service import ForecastingService
from forecasting.application.services.model_selector import ModelSelector

# Driven Adapters
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

# Driving Adapter
from forecasting.adapters.driving.api.forecast_controller import ForecastController


def create_app() -> FastAPI:
    """Application factory assembling dependencies according to Hexagonal Architecture."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Hexagonal Architecture AI Forecasting Module for JCD Commerce SaaS ERP.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 1. Instantiate Driven Adapters
    data_loader = ERPDataLoaderAdapter()
    data_cleaner = StandardDataCleanerAdapter()
    
    preparators = {
        "Prophet": ProphetDataPreparatorAdapter(),
        "XGBoost": XGBoostDataPreparatorAdapter(),
    }

    models = {
        "Prophet": ProphetModelAdapter(),
        "XGBoost": XGBoostModelAdapter(),
    }

    evaluators = [
        MAEEvaluatorAdapter(),
        RMSEEvaluatorAdapter(),
        MAPEEvaluatorAdapter(),
        R2EvaluatorAdapter(),
    ]

    model_selector = ModelSelector(primary_metric=settings.primary_metric)

    # 2. Assemble Application Core Service
    forecasting_service = ForecastingService(
        data_loader=data_loader,
        data_cleaner=data_cleaner,
        preparators=preparators,
        models=models,
        evaluators=evaluators,
        model_selector=model_selector,
    )

    # 3. Mount Driving Adapter (FastAPI Controller)
    controller = ForecastController(use_case=forecasting_service)
    app.include_router(controller.router)

    @app.get("/health", tags=["Health"])
    def health_check():
        return {
            "status": "healthy",
            "service": settings.app_name,
            "version": settings.app_version,
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("forecasting.main:app", host="0.0.0.0", port=8000, reload=True)
