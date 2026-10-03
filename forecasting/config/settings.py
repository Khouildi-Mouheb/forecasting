from pydantic import BaseModel


class Settings(BaseModel):
    """Application and pipeline configuration settings."""

    app_name: str = "JCD ERP Forecasting Microservice"
    app_version: str = "1.0.0"
    default_horizon: int = 14
    primary_metric: str = "MAE"
    api_prefix: str = "/api/v1"


settings = Settings()
