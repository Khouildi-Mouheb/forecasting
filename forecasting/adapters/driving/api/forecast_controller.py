from fastapi import APIRouter, HTTPException, Depends
from forecasting.application.ports.driving.forecasting_use_case import ForecastingUseCase
from forecasting.domain.forecast_request import ForecastRequest
from forecasting.adapters.driving.api.schemas import (
    ForecastRequestSchema,
    ForecastResponseSchema,
    ConfidenceIntervalSchema,
)


class ForecastController:
    """
    Driving Adapter: Implements ForecastController from class diagram.
    Handles HTTP requests, maps them into domain ForecastRequest entities,
    invokes the ForecastingUseCase driving port, and returns HTTP responses.
    """

    def __init__(self, use_case: ForecastingUseCase):
        self.use_case = use_case
        self.router = APIRouter(prefix="/api/v1", tags=["Forecasting"])
        self._register_routes()

    def _register_routes(self):
        @self.router.post("/forecast", response_model=ForecastResponseSchema)
        def forecast_endpoint(payload: ForecastRequestSchema):
            return self.handleForecastRequest(payload)

    def handleForecastRequest(self, httpRequest: ForecastRequestSchema) -> ForecastResponseSchema:
        """Processes incoming HTTP request and returns structured forecast response."""
        try:
            domain_request = ForecastRequest(
                itemId=httpRequest.itemId,
                horizon=httpRequest.horizon,
                modelType=httpRequest.modelType,
                history=[p.model_dump() for p in httpRequest.history],
            )

            result = self.use_case.generateForecast(domain_request)

            conf_int = None
            if result.confidenceInterval:
                conf_int = ConfidenceIntervalSchema(
                    lower=result.confidenceInterval.get("lower", []),
                    upper=result.confidenceInterval.get("upper", []),
                )

            return ForecastResponseSchema(
                itemId=result.itemId,
                predictedValues=result.predictedValues,
                confidenceInterval=conf_int,
                modelUsed=result.modelUsed,
                generatedAt=result.generatedAt,
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")
