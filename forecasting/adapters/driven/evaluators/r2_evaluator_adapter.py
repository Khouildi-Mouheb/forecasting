from typing import Sequence
from sklearn.metrics import r2_score

from forecasting.application.ports.driven.evaluator_port import EvaluatorPort
from forecasting.domain.evaluation_result import EvaluationResult


class R2EvaluatorAdapter(EvaluatorPort):
    """Driven Adapter: Implements EvaluatorPort for Coefficient of Determination (R2)."""

    def evaluate(self, actual: Sequence[float], predicted: Sequence[float]) -> EvaluationResult:
        if not actual or not predicted or len(actual) < 2:
            return EvaluationResult(metricName="R2", value=0.0)

        min_len = min(len(actual), len(predicted))
        try:
            score = float(r2_score(actual[:min_len], predicted[:min_len]))
        except Exception:
            score = 0.0

        return EvaluationResult(metricName="R2", value=round(score, 4))
