from typing import Dict, List, Optional
from forecasting.domain.forecast_result import ForecastResult
from forecasting.domain.evaluation_result import EvaluationResult


class ModelSelector:
    """Domain logic responsible for choosing the champion model based on validation metrics."""

    def __init__(self, primary_metric: str = "MAE"):
        self.primary_metric = primary_metric.upper()

    def selectBest(
        self,
        candidate_results: Dict[str, ForecastResult],
        evaluations: Dict[str, List[EvaluationResult]],
    ) -> ForecastResult:
        """
        Selects the best performing ForecastResult from candidates based on evaluation results.
        
        Args:
            candidate_results: Mapping of model_name -> ForecastResult
            evaluations: Mapping of model_name -> List of EvaluationResult
        
        Returns:
            The winning ForecastResult.
        """
        if not candidate_results:
            raise ValueError("No candidate model results provided to ModelSelector.")

        if len(candidate_results) == 1:
            return next(iter(candidate_results.values()))

        best_model_name: Optional[str] = None
        best_score: Optional[float] = None

        # For error metrics, lower is better. For R2, higher is better.
        lower_is_better = self.primary_metric in {"MAE", "RMSE", "MAPE", "SMAPE"}

        for model_name, results in evaluations.items():
            if model_name not in candidate_results:
                continue

            metric_result = next(
                (r for r in results if r.metricName.upper() == self.primary_metric),
                None,
            )

            if metric_result is None:
                # Fallback to the first available metric if primary is not found
                if results:
                    metric_result = results[0]
                else:
                    continue

            score = metric_result.value

            if best_score is None:
                best_score = score
                best_model_name = model_name
            else:
                if lower_is_better and score < best_score:
                    best_score = score
                    best_model_name = model_name
                elif not lower_is_better and score > best_score:
                    best_score = score
                    best_model_name = model_name

        if best_model_name is not None and best_model_name in candidate_results:
            return candidate_results[best_model_name]

        return next(iter(candidate_results.values()))
