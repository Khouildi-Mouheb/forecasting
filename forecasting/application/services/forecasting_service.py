from typing import Dict, List, Optional
import numpy as np

from forecasting.application.ports.driving.forecasting_use_case import ForecastingUseCase
from forecasting.application.ports.driven.data_loader_port import DataLoaderPort
from forecasting.application.ports.driven.data_cleaner_port import DataCleanerPort
from forecasting.application.ports.driven.data_preparator_port import DataPreparatorPort
from forecasting.application.ports.driven.model_port import ModelPort
from forecasting.application.ports.driven.evaluator_port import EvaluatorPort
from forecasting.application.services.model_selector import ModelSelector

from forecasting.domain.forecast_request import ForecastRequest
from forecasting.domain.forecast_result import ForecastResult
from forecasting.domain.evaluation_result import EvaluationResult
from forecasting.domain.cleaned_data import CleanedData


class ForecastingService(ForecastingUseCase):
    """
    Application Core: Orchestrates data loading, cleaning, feature preparation,
    model prediction, evaluation, and selection according to Hexagonal Architecture.
    """

    def __init__(
        self,
        data_loader: DataLoaderPort,
        data_cleaner: DataCleanerPort,
        preparators: Dict[str, DataPreparatorPort],
        models: Dict[str, ModelPort],
        evaluators: List[EvaluatorPort],
        model_selector: Optional[ModelSelector] = None,
    ):
        self.data_loader = data_loader
        self.data_cleaner = data_cleaner
        self.preparators = preparators
        self.models = models
        self.evaluators = evaluators
        self.model_selector = model_selector or ModelSelector()

    def generateForecast(self, request: ForecastRequest) -> ForecastResult:
        """
        Executes the end-to-end forecasting pipeline.
        """
        # 1. Ingest Raw Data
        raw_data = self.data_loader.load(request)

        # 2. Clean and Validate Data
        cleaned_data = self.data_cleaner.clean(raw_data)

        if not cleaned_data.points or len(cleaned_data.points) < 5:
            raise ValueError(
                f"Insufficient historical data points for item '{request.itemId}'. Minimum 5 required."
            )

        # 3. Model execution
        # If a specific model is requested and available, use it directly
        if request.modelType and request.modelType in self.models:
            model_key = request.modelType
            preparator = self.preparators[model_key]
            prepared_data = preparator.prepare(cleaned_data)
            model = self.models[model_key]
            result = model.predict(prepared_data, request.horizon)
            result.itemId = request.itemId
            return result

        # Otherwise, run available candidate models, evaluate on holdout set, and select best
        candidate_results: Dict[str, ForecastResult] = {}
        evaluations: Dict[str, List[EvaluationResult]] = {}

        # Use an evaluation holdout window to score models
        train_data, val_data = self._split_cleaned_data(cleaned_data, val_size=min(request.horizon, max(3, len(cleaned_data.points) // 5)))

        for model_key, model in self.models.items():
            preparator = self.preparators.get(model_key)
            if not preparator:
                continue

            try:
                # Predict on full data for the final response
                full_prepared = preparator.prepare(cleaned_data)
                final_result = model.predict(full_prepared, request.horizon)
                final_result.itemId = request.itemId
                candidate_results[model_key] = final_result

                # Train on train_data and evaluate on val_data if sufficient points
                if len(train_data.points) >= 5 and len(val_data.points) >= 1:
                    train_prepared = preparator.prepare(train_data)
                    val_result = model.predict(train_prepared, len(val_data.points))
                    
                    actual_vals = [p.quantity for p in val_data.points]
                    pred_vals = val_result.predictedValues[:len(actual_vals)]

                    eval_list = [
                        evaluator.evaluate(actual_vals, pred_vals)
                        for evaluator in self.evaluators
                    ]
                    evaluations[model_key] = eval_list
            except Exception as e:
                # Log or handle individual model failures gracefully
                continue

        if not candidate_results:
            raise RuntimeError("All candidate forecasting models failed to generate predictions.")

        # 4. Select champion model
        champion_result = self.model_selector.selectBest(candidate_results, evaluations)
        champion_result.itemId = request.itemId
        return champion_result

    def _split_cleaned_data(self, cleaned: CleanedData, val_size: int):
        val_size = max(1, min(val_size, len(cleaned.points) - 4))
        train_points = cleaned.points[:-val_size]
        val_points = cleaned.points[-val_size:]
        return CleanedData(points=train_points), CleanedData(points=val_points)
