"""
FastAPI scoring service.

    uvicorn src.serving.app:app --port 8000

GET  /                               ->  demo page: type an applicant ID, see the result
POST /score  {"sk_id_curr": 100002}  ->  probability, decision, top risk factors
POST /score-custom {"ext_source_2": 0.6, ...} -> the same for an invented applicant (no record looked up)
GET  /health                         ->  liveness plus the model version in use
GET  /model-info                     ->  threshold and the feature list

Requires the Feast online store to be populated (`feast materialize-incremental`).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.serving.page import PAGE
from src.serving.scoring import ApplicantNotFound, ScoringService


class ScoreRequest(BaseModel):
    sk_id_curr: int = Field(..., description="Applicant ID (SK_ID_CURR)", gt=0)


class CustomApplicant(BaseModel):
    """An invented applicant. Leave a field out (or null) for 'unknown'."""
    ext_source_1: float | None = Field(None, ge=0, le=1, description="External credit score 1, 0 to 1")
    ext_source_2: float | None = Field(None, ge=0, le=1, description="External credit score 2, 0 to 1")
    ext_source_3: float | None = Field(None, ge=0, le=1, description="External credit score 3, 0 to 1")
    age_years: float | None = Field(None, ge=18, le=100)
    employment_years: float | None = Field(None, ge=0, le=60)
    income_total: float | None = Field(None, ge=0, le=1e9)
    credit_amount: float | None = Field(None, ge=0, le=1e9)
    goods_price: float | None = Field(None, ge=0, le=1e9)
    annuity_amount: float | None = Field(None, ge=0, le=1e9)
    education_level: int | None = Field(None, ge=0, le=4, description="0 lower secondary ... 4 academic degree")


class RiskFactor(BaseModel):
    feature: str
    reason: str
    contribution: float


class Factor(BaseModel):
    feature: str
    label: str
    value_text: str
    contribution: float
    direction: str


class ScoreResponse(BaseModel):
    sk_id_curr: int | None = None  # None for an invented applicant
    default_probability: float
    decision: str
    threshold: float
    model_version: str
    top_risk_factors: list[RiskFactor]
    factors: list[Factor] = []
    riskier_than_pct: int | None = None
    risk_band: int | None = None
    average_default_probability: float | None = None


def build_default_service() -> ScoringService:
    import shap
    from feast import FeatureStore

    from src.common import config
    from src.serving.registry import load_champion

    model, version, threshold = load_champion()
    reference = None
    try:  # percentile / risk band context comes from the batch-scored applicants, if that has been run
        from deltalake import DeltaTable

        if DeltaTable.is_deltatable(config.SCORED_TABLE):
            reference = DeltaTable(config.SCORED_TABLE).to_pandas(columns=["default_probability"])["default_probability"]
    except Exception:  # the page still works without it
        reference = None
    store = FeatureStore(repo_path=str(config.FEAST_REPO_DIR))
    return ScoringService(
        model=model, threshold=threshold, model_version=version, store=store,
        feature_service=store.get_feature_service(config.FEAST_FEATURE_SERVICE),
        explainer=shap.TreeExplainer(model), reference_scores=reference,
    )


def create_app(service: ScoringService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or build_default_service()
        yield

    app = FastAPI(title="Credit risk scoring", version="1.0", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home():
        return PAGE

    @app.get("/health")
    def health():
        info = app.state.service.info()
        return {"status": "ok", "model_version": info["model_version"], "threshold": info["threshold"]}

    @app.get("/model-info")
    def model_info():
        return app.state.service.info()

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest):
        try:
            return app.state.service.score(req.sk_id_curr)
        except ApplicantNotFound:
            raise HTTPException(
                status_code=404,
                detail=(f"No features for applicant {req.sk_id_curr} in the online store. "
                        "Check the ID, and that `feast materialize-incremental` has been run."),
            )

    @app.post("/score-custom", response_model=ScoreResponse)
    def score_custom(req: CustomApplicant):
        return app.state.service.score_custom(req.model_dump())

    return app


app = create_app()
