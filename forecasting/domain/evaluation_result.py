from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    metricName: str
    value: float
