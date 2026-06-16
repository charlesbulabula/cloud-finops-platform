"""
AWS Cost Explorer Ingester — fetches daily/monthly costs grouped by service
and tag, handles pagination, stores results in TimescaleDB or writes Prometheus
metrics via prometheus_client.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Iterator

import boto3
import psycopg2
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logger = logging.getLogger(__name__)

COST_EXPLORER_REGION = "us-east-1"  # CE is a global service, always us-east-1


@dataclass
class CostRecord:
    date: str
    service: str
    tag_key: str
    tag_value: str
    amount: float
    unit: str
    account_id: str = ""


@dataclass
class IngestionConfig:
    group_by_tags: list[str] = field(default_factory=lambda: ["Team", "Environment", "CostCenter"])
    granularity: str = "DAILY"  # DAILY | MONTHLY
    lookback_days: int = 30
    timescaledb_dsn: str = field(default_factory=lambda: os.environ.get("TIMESCALEDB_DSN", ""))
    prometheus_gateway: str = field(default_factory=lambda: os.environ.get("PROMETHEUS_GATEWAY", ""))
    prometheus_job: str = "aws-cost-explorer"


def _date_range(lookback_days: int) -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=lookback_days)
    return start.isoformat(), end.isoformat()


def _build_group_by(tag_keys: list[str]) -> list[dict]:
    groups = [{"Type": "DIMENSION", "Key": "SERVICE"}]
    for tag in tag_keys[:1]:  # CE allows max 2 group-by dimensions
        groups.append({"Type": "TAG", "Key": tag})
    return groups


def _paginate_cost_results(
    ce_client: Any,
    start: str,
    end: str,
    granularity: str,
    group_by: list[dict],
) -> Iterator[dict]:
    kwargs: dict = {
        "TimePeriod": {"Start": start, "End": end},
        "Granularity": granularity,
        "Metrics": ["UnblendedCost"],
        "GroupBy": group_by,
    }
    while True:
        response = ce_client.get_cost_and_usage(**kwargs)
        for result in response.get("ResultsByTime", []):
            yield result
        next_token = response.get("NextPageToken")
        if not next_token:
            break
        kwargs["NextPageToken"] = next_token


def fetch_costs(
    config: IngestionConfig,
    session: boto3.Session | None = None,
) -> list[CostRecord]:
    """Fetches cost data from AWS Cost Explorer and returns a list of CostRecords."""
    if session is None:
        session = boto3.Session()

    ce = session.client("ce", region_name=COST_EXPLORER_REGION)
    start, end = _date_range(config.lookback_days)
    group_by = _build_group_by(config.group_by_tags)

    records: list[CostRecord] = []
    tag_key = config.group_by_tags[0] if config.group_by_tags else "Team"

    for result in _paginate_cost_results(ce, start, end, config.granularity, group_by):
        period_start = result["TimePeriod"]["Start"]
        for group in result.get("Groups", []):
            keys = group.get("Keys", ["", ""])
            service = keys[0] if len(keys) > 0 else "Unknown"
            tag_raw = keys[1] if len(keys) > 1 else ""
            tag_value = tag_raw.replace(f"{tag_key}$", "") if "$" in tag_raw else tag_raw

            metrics = group.get("Metrics", {})
            cost_data = metrics.get("UnblendedCost", {})
            amount = float(cost_data.get("Amount", 0))
            unit = cost_data.get("Unit", "USD")

            records.append(
                CostRecord(
                    date=period_start,
                    service=service,
                    tag_key=tag_key,
                    tag_value=tag_value or "untagged",
                    amount=round(amount, 6),
                    unit=unit,
                )
            )

    logger.info("Fetched %d cost records from %s to %s", len(records), start, end)
    return records


def write_to_timescaledb(records: list[CostRecord], dsn: str) -> None:
    """Upserts cost records into TimescaleDB hypertable `aws_costs`."""
    if not records:
        return

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS aws_costs (
                    time        TIMESTAMPTZ NOT NULL,
                    service     TEXT NOT NULL,
                    tag_key     TEXT,
                    tag_value   TEXT,
                    amount      DOUBLE PRECISION,
                    unit        TEXT,
                    account_id  TEXT
                )
                """
            )
            # Ensure hypertable
            cur.execute(
                "SELECT create_hypertable('aws_costs','time',if_not_exists=>TRUE)"
            )
            rows = [
                (r.date, r.service, r.tag_key, r.tag_value, r.amount, r.unit, r.account_id)
                for r in records
            ]
            cur.executemany(
                """
                INSERT INTO aws_costs (time, service, tag_key, tag_value, amount, unit, account_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                rows,
            )
        conn.commit()
        logger.info("Wrote %d records to TimescaleDB", len(rows))
    finally:
        conn.close()


def write_prometheus_metrics(
    records: list[CostRecord],
    gateway: str,
    job: str,
) -> None:
    """Pushes cost metrics to a Prometheus Pushgateway."""
    registry = CollectorRegistry()
    gauge = Gauge(
        "aws_daily_cost_usd",
        "AWS daily cost in USD grouped by service and tag",
        labelnames=["service", "tag_key", "tag_value", "date"],
        registry=registry,
    )
    for r in records:
        gauge.labels(
            service=r.service,
            tag_key=r.tag_key,
            tag_value=r.tag_value,
            date=r.date,
        ).set(r.amount)

    push_to_gateway(gateway, job=job, registry=registry)
    logger.info("Pushed %d metrics to Prometheus gateway at %s", len(records), gateway)


def ingest(config: IngestionConfig, session: boto3.Session | None = None) -> list[CostRecord]:
    """Main entrypoint: fetch costs and dispatch to configured sinks."""
    records = fetch_costs(config, session)

    if config.timescaledb_dsn:
        write_to_timescaledb(records, config.timescaledb_dsn)

    if config.prometheus_gateway:
        write_prometheus_metrics(records, config.prometheus_gateway, config.prometheus_job)

    if not config.timescaledb_dsn and not config.prometheus_gateway:
        logger.warning("No output sink configured; records are returned but not persisted")

    return records

# _r 20260616103912-8b04ee67
