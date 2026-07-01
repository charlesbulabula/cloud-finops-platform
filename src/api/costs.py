"""
FastAPI router for FinOps cost APIs.
  GET  /costs/summary    — cost breakdown by period and grouping
  GET  /costs/anomalies  — spend anomalies with sigma scores (last 30 days)
  GET  /costs/forecast   — linear regression forecast for next N days
  GET  /costs/rightsizing — EC2/RDS rightsizing recommendations
"""

import logging
import statistics
from datetime import date, timedelta
from typing import Any, Literal

import boto3
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel

from src.anomaly.detector import AnomalyAlert, detect_anomalies
from src.ingestion.cost_explorer import IngestionConfig, fetch_costs
from src.rightsizing.optimizer import RightsizingRecommendation, analyze_rightsizing

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/costs", tags=["FinOps Costs"])

# ---------------------------------------------------------------------------
# Shared boto3 session dependency
# ---------------------------------------------------------------------------


def get_session() -> boto3.Session:
    return boto3.Session()


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class ServiceCost(BaseModel):
    service: str
    cost: float
    currency: str = "USD"


class CostSummaryResponse(BaseModel):
    period: str
    total_cost: float
    currency: str = "USD"
    group_by: str
    item_count: int
    breakdown: list[ServiceCost]


class AnomalyItem(BaseModel):
    alert_date: str
    actual_spend: float
    expected_spend: float
    threshold: float
    sigma: float
    top_services: list[tuple[str, float]]


class ForecastPoint(BaseModel):
    date: str
    predicted_cost: float
    lower_bound: float
    upper_bound: float


class ForecastResponse(BaseModel):
    history_days: int
    forecast_days: int
    method: str = "linear_regression"
    currency: str = "USD"
    points: list[ForecastPoint]


class RightsizingItem(BaseModel):
    resource_id: str
    resource_type: str
    current_type: str
    recommended_type: str
    cpu_p95: float
    memory_p95: float | None
    reason: str
    estimated_savings_pct: float


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _parse_period_to_days(period: str) -> int:
    """Parse '7d', '30d', 'mtd' into number of lookback days."""
    period = period.lower().strip()
    if period == "mtd":
        today = date.today()
        return today.day  # days since month start
    if period.endswith("d"):
        return int(period[:-1])
    if period.endswith("m"):
        return int(period[:-1]) * 30
    raise ValueError(
        f"Unsupported period '{period}'. Use '7d', '30d', 'mtd', or 'Nm'."
    )


def _linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Return (slope, intercept) via ordinary least squares."""
    n = len(xs)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_xy = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    slope = ss_xy / ss_xx if ss_xx != 0 else 0.0
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _compute_residual_std(
    xs: list[float], ys: list[float], slope: float, intercept: float
) -> float:
    """Compute standard deviation of regression residuals."""
    residuals = [ys[i] - (slope * xs[i] + intercept) for i in range(len(xs))]
    if len(residuals) < 2:
        return 0.0
    return statistics.stdev(residuals)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/summary", response_model=CostSummaryResponse)
async def get_cost_summary(
    period: str = Query("7d", description="Lookback period: '7d', '30d', 'mtd'"),
    group_by: Literal["service", "tag", "region"] = Query(
        "service", description="Grouping dimension"
    ),
    session: boto3.Session = Depends(get_session),
) -> CostSummaryResponse:
    """Return aggregated AWS costs for the specified period and grouping."""
    try:
        days = _parse_period_to_days(period)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    tag_map: dict[str, list[str]] = {
        "service": [],
        "tag": ["Team"],
        "region": [],
    }

    cfg = IngestionConfig(
        group_by_tags=tag_map.get(group_by, []),
        granularity="DAILY",
        lookback_days=days,
    )

    try:
        records = fetch_costs(cfg, session)
    except Exception as exc:
        logger.exception("Cost Explorer fetch failed")
        raise HTTPException(status_code=502, detail=f"Cost Explorer error: {exc}")

    totals: dict[str, float] = {}
    for r in records:
        key = r.tag_value if group_by == "tag" else r.service
        totals[key] = totals.get(key, 0.0) + r.amount

    breakdown = sorted(
        [ServiceCost(service=k, cost=round(v, 2)) for k, v in totals.items()],
        key=lambda s: s.cost,
        reverse=True,
    )
    total_cost = round(sum(totals.values()), 2)
    start_date = (date.today() - timedelta(days=days)).isoformat()
    end_date = date.today().isoformat()

    return CostSummaryResponse(
        period=f"{start_date}/{end_date}",
        total_cost=total_cost,
        group_by=group_by,
        item_count=len(breakdown),
        breakdown=breakdown,
    )


@router.get("/anomalies", response_model=list[AnomalyItem])
async def get_anomalies(
    session: boto3.Session = Depends(get_session),
) -> list[AnomalyItem]:
    """Detect spend anomalies over the last 30 days using a 2-sigma rolling baseline."""
    try:
        alerts: list[AnomalyAlert] = detect_anomalies(
            session=session, dry_run=True
        )
    except Exception as exc:
        logger.exception("Anomaly detection failed")
        raise HTTPException(status_code=502, detail=str(exc))

    return [
        AnomalyItem(
            alert_date=a.alert_date,
            actual_spend=a.actual_spend,
            expected_spend=a.expected_spend,
            threshold=a.threshold,
            sigma=a.sigma,
            top_services=a.top_services,
        )
        for a in alerts
    ]


@router.get("/forecast", response_model=ForecastResponse)
async def get_cost_forecast(
    days: int = Query(30, ge=1, le=365, description="Number of days to forecast"),
    history_days: int = Query(60, ge=14, le=180, description="Historical window for regression"),
    session: boto3.Session = Depends(get_session),
) -> ForecastResponse:
    """
    Forecast future spend using linear regression on historical daily costs.
    Returns predicted cost with 95% confidence bounds per day.
    """
    cfg = IngestionConfig(granularity="DAILY", lookback_days=history_days)
    try:
        records = fetch_costs(cfg, session)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    if not records:
        raise HTTPException(status_code=404, detail="No historical cost data available.")

    # Aggregate by date
    daily: dict[str, float] = {}
    for r in records:
        daily[r.date] = daily.get(r.date, 0.0) + r.amount

    sorted_dates = sorted(daily.keys())
    xs = list(range(len(sorted_dates)))
    ys = [daily[d] for d in sorted_dates]

    slope, intercept = _linear_regression(xs, ys)
    residual_std = _compute_residual_std(xs, ys, slope, intercept)
    confidence_delta = 1.96 * residual_std  # 95% interval

    last_x = xs[-1] if xs else -1
    last_date = date.fromisoformat(sorted_dates[-1]) if sorted_dates else date.today()

    points: list[ForecastPoint] = []
    for i in range(1, days + 1):
        x_val = last_x + i
        predicted = max(0.0, slope * x_val + intercept)
        forecast_date = (last_date + timedelta(days=i)).isoformat()
        points.append(
            ForecastPoint(
                date=forecast_date,
                predicted_cost=round(predicted, 2),
                lower_bound=round(max(0.0, predicted - confidence_delta), 2),
                upper_bound=round(predicted + confidence_delta, 2),
            )
        )

    return ForecastResponse(
        history_days=history_days,
        forecast_days=days,
        points=points,
    )


@router.get("/rightsizing", response_model=list[RightsizingItem])
async def get_rightsizing_recommendations(
    regions: str = Query("us-east-1", description="Comma-separated AWS regions"),
    session: boto3.Session = Depends(get_session),
) -> list[RightsizingItem]:
    """Return EC2 and RDS rightsizing recommendations based on 14-day CloudWatch utilization."""
    region_list = [r.strip() for r in regions.split(",") if r.strip()]
    try:
        recs: list[RightsizingRecommendation] = analyze_rightsizing(
            regions=region_list, session=session
        )
    except Exception as exc:
        logger.exception("Rightsizing analysis failed")
        raise HTTPException(status_code=502, detail=str(exc))

    return [
        RightsizingItem(
            resource_id=r.resource_id,
            resource_type=r.resource_type,
            current_type=r.current_type,
            recommended_type=r.recommended_type,
            cpu_p95=r.cpu_p95,
            memory_p95=r.memory_p95,
            reason=r.reason,
            estimated_savings_pct=r.estimated_savings_pct,
        )
        for r in recs
    ]

# _r 20260701135904-5e52085a
