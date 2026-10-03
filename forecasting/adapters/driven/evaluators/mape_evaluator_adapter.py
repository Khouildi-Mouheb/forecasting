from typing import Sequence
import numpy as np

from forecasting.application.ports.driven.evaluator_port import EvaluatorPort
from forecasting.domain.evaluation_result import EvaluationResult


class MAPEEvaluatorAdapter(EvaluatorPort):
    """
    Driven Adapter: Implements EvaluatorPort for Mean Absolute Percentage Error (MAPE).
    Uses safe epsilon handling to avoid division by zero when actual values are 0.
    """

    def evaluate(self, actual: Sequence[float], predicted: Sequence[float]) -> EvaluationResult:
        if not actual or not predicted:
            return EvaluationResult(metricName="MAPE", value=float("inf"))

        min_len = min(len(actual), len(predicted))
        y_true = np.array(actual[:min_len], dtype=float)
        y_pred = np.array(predicted[:min_len], dtype=float)

        denominator = np.where(y_true == 0, 1.0, np.abs(y_true))
        percentage_errors = np.abs((y_true - y_pred) / denominator) * 100.0
        score = float(np.mean(percentage_errors))

        return EvaluationResult(metricName="MAPE", value=round(score, 4))
