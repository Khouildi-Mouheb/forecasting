from typing import Sequence
from sklearn.metrics import mean_absolute_error

from forecasting.application.ports.driven.evaluator_port import EvaluatorPort
from forecasting.domain.evaluation_result import EvaluationResult


class MAEEvaluatorAdapter(EvaluatorPort):
    """Driven Adapter: Implements EvaluatorPort for Mean Absolute Error (MAE)."""

    def evaluate(self, actual: Sequence[float], predicted: Sequence[float]) -> EvaluationResult:
        if not actual or not predicted:
            return EvaluationResult(metricName="MAE", value=float("inf"))

        min_len = min(len(actual), len(predicted))
        score = float(mean_absolute_error(actual[:min_len], predicted[:min_len]))
        return EvaluationResult(metricName="MAE", value=round(score, 4))
