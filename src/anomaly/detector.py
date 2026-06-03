"""
Spend Anomaly Detector — uses a rolling 30-day average + 2-sigma threshold to
detect daily spend anomalies, and sends Slack notifications with a per-service
cost breakdown when triggered.
"""

import logging
import os
import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import boto3
import requests

logger = logging.getLogger(__name__)

SLACK_WEBHOOK_ENV = "SLACK_WEBHOOK_URL"
SIGMA_MULTIPLIER = 2.0
ROLLING_WINDOW_DAYS = 30
MIN_DAILY_SPEND_THRESHOLD = 1.0  # USD — ignore noise below this


@dataclass
class DailySpend:
    date: str
    total: float
    by_service: dict[str, float]


@dataclass
class AnomalyAlert:
    alert_date: str
    actual_spend: float
    expected_spend: float
    threshold: float
    sigma: float
    top_services: list[tuple[str, float]]


def _fetch_daily_spend_history(
    ce_client: Any,
    lookback_days: int = ROLLING_WINDOW_DAYS + 1,
) -> list[DailySpend]:
    """Fetches daily unblended costs grouped by service for the past N days."""
    end = date.today()
    start = end - timedelta(days=lookback_days)

    response = ce_client.get_cost_and_usage(
        TimePeriod={"Start": start.isoformat(), "End": end.isoformat()},
        Granularity="DAILY",
        Metrics=["UnblendedCost"],
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )

    daily_records: dict[str, DailySpend] = {}
    for result in response.get("ResultsByTime", []):
        period = result["TimePeriod"]["Start"]
        by_service: dict[str, float] = {}
        for group in result.get("Groups", []):
            service = group["Keys"][0]
            amount = float(group["Metrics"]["UnblendedCost"]["Amount"])
            if amount > 0:
                by_service[service] = round(amount, 4)
        total = sum(by_service.values())
        daily_records[period] = DailySpend(date=period, total=round(total, 4), by_service=by_service)

    return sorted(daily_records.values(), key=lambda d: d.date)


def _compute_threshold(historical_totals: list[float]) -> tuple[float, float]:
    """
    Returns (mean, threshold) where threshold = mean + SIGMA_MULTIPLIER * stdev.
    Requires at least 7 data points; falls back to mean * 1.5 if fewer.
    """
    if len(historical_totals) < 7:
        mean = statistics.mean(historical_totals) if historical_totals else 0.0
        return mean, mean * 1.5

    mean = statistics.mean(historical_totals)
    stdev = statistics.stdev(historical_totals)
    threshold = mean + SIGMA_MULTIPLIER * stdev
    return mean, threshold


def _format_slack_message(alert: AnomalyAlert) -> dict:
    top_services_text = "\n".join(
        f"  • *{svc}*: ${amt:.2f}" for svc, amt in alert.top_services[:5]
    )
    return {
        "text": f":rotating_light: *AWS Spend Anomaly Detected* ({alert.alert_date})",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"Spend Anomaly: {alert.alert_date}"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Actual Spend:*\n${alert.actual_spend:.2f}"},
                    {"type": "mrkdwn", "text": f"*Expected (30d avg):*\n${alert.expected_spend:.2f}"},
                    {"type": "mrkdwn", "text": f"*Threshold ({SIGMA_MULTIPLIER}σ):*\n${alert.threshold:.2f}"},
                    {"type": "mrkdwn", "text": f"*Overage:*\n${alert.actual_spend - alert.threshold:.2f}"},
                ],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Top Services:*\n{top_services_text}"},
            },
        ],
    }


def _send_slack_alert(alert: AnomalyAlert, webhook_url: str) -> None:
    payload = _format_slack_message(alert)
    resp = requests.post(webhook_url, json=payload, timeout=10)
    if resp.status_code != 200:
        logger.error("Slack notification failed: %s %s", resp.status_code, resp.text)
    else:
        logger.info("Slack alert sent for anomaly on %s", alert.alert_date)


def detect_anomalies(
    session: boto3.Session | None = None,
    slack_webhook: str | None = None,
    dry_run: bool = False,
) -> list[AnomalyAlert]:
    """
    Main detection function. Fetches 31 days of spend, uses first 30 days as
    baseline, evaluates yesterday's spend against the threshold.
    """
    if session is None:
        session = boto3.Session()
    if slack_webhook is None:
        slack_webhook = os.environ.get(SLACK_WEBHOOK_ENV)

    ce = session.client("ce", region_name="us-east-1")
    history = _fetch_daily_spend_history(ce, lookback_days=ROLLING_WINDOW_DAYS + 1)

    if len(history) < 2:
        logger.warning("Insufficient history for anomaly detection (%d days)", len(history))
        return []

    # Baseline = all days except the most recent
    baseline = history[:-1]
    today_record = history[-1]

    if today_record.total < MIN_DAILY_SPEND_THRESHOLD:
        logger.info("Daily spend $%.2f below noise floor, skipping", today_record.total)
        return []

    baseline_totals = [d.total for d in baseline]
    mean, threshold = _compute_threshold(baseline_totals)

    alerts: list[AnomalyAlert] = []
    if today_record.total > threshold:
        top_services = sorted(today_record.by_service.items(), key=lambda x: x[1], reverse=True)
        sigma = (today_record.total - mean) / (statistics.stdev(baseline_totals) or 1)
        alert = AnomalyAlert(
            alert_date=today_record.date,
            actual_spend=today_record.total,
            expected_spend=round(mean, 2),
            threshold=round(threshold, 2),
            sigma=round(sigma, 2),
            top_services=top_services,
        )
        alerts.append(alert)
        logger.warning(
            "Anomaly detected on %s: $%.2f actual vs $%.2f threshold (%.1fσ)",
            today_record.date,
            today_record.total,
            threshold,
            sigma,
        )

        if slack_webhook and not dry_run:
            _send_slack_alert(alert, slack_webhook)
        elif dry_run:
            logger.info("[DRY RUN] Would send Slack alert: %s", _format_slack_message(alert))
    else:
        logger.info(
            "No anomaly on %s: $%.2f (threshold=$%.2f)", today_record.date, today_record.total, threshold
        )

    return alerts

# _r 20260521153608-ccfc054f
