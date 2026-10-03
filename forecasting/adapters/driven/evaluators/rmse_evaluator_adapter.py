from typing import Sequence
import numpy as np
from sklearn.metrics import root_mean_squared_error

from forecasting.application.ports.driven.evaluator_port import EvaluatorPort
from forecasting.domain.evaluation_result import EvaluationResult


class RMSEEvaluatorAdapter(EvaluatorPort):
    """Driven Adapter: Implements EvaluatorPort for Root Mean Squared Error (RMSE)."""

    def evaluate(self, actual: Sequence[float], predicted: Sequence[float]) -> EvaluationResult:
        if not actual or not predicted:
            return EvaluationResult(metricName="RMSE", value=float("inf"))

        min_len = min(len(actual), len(predicted))
        score = float(root_mean_squared_error(actual[:min_len], predicted[:min_len]))
        return EvaluationResult(metricName="RMSE", value=round(score, 4))
