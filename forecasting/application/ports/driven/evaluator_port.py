from abc import ABC, abstractmethod
from typing import Sequence
from forecasting.domain.evaluation_result import EvaluationResult


class EvaluatorPort(ABC):
    """Driven Port: Interface for error metric evaluators (MAE, RMSE, MAPE, R2)."""

    @abstractmethod
    def evaluate(self, actual: Sequence[float], predicted: Sequence[float]) -> EvaluationResult:
        """Computes evaluation metric between ground truth and predicted sequences."""
        pass
