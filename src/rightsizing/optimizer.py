"""
EC2/RDS Rightsizing Analyzer — uses CloudWatch GetMetricStatistics for CPU and
memory utilization over 14 days, compares to current instance type, and
recommends downsizing when p95 CPU < 20%.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 14
CPU_DOWNSIZE_THRESHOLD_PCT = 20.0  # p95 CPU below this → recommend downsize
MEMORY_DOWNSIZE_THRESHOLD_PCT = 30.0

# EC2 instance family downsize map (current → recommended)
EC2_DOWNSIZE_MAP: dict[str, str] = {
    "t3.large": "t3.medium",
    "t3.medium": "t3.small",
    "t3.xlarge": "t3.large",
    "m5.large": "t3.large",
    "m5.xlarge": "m5.large",
    "m5.2xlarge": "m5.xlarge",
    "m5.4xlarge": "m5.2xlarge",
    "c5.large": "t3.large",
    "c5.xlarge": "c5.large",
    "c5.2xlarge": "c5.xlarge",
    "r5.large": "m5.large",
    "r5.xlarge": "r5.large",
}

RDS_DOWNSIZE_MAP: dict[str, str] = {
    "db.t3.large": "db.t3.medium",
    "db.t3.medium": "db.t3.small",
    "db.m5.large": "db.t3.large",
    "db.m5.xlarge": "db.m5.large",
    "db.m5.2xlarge": "db.m5.xlarge",
    "db.r5.large": "db.m5.large",
    "db.r5.xlarge": "db.r5.large",
}


@dataclass
class UtilizationStats:
    resource_id: str
    resource_type: str  # EC2 | RDS
    instance_type: str
    region: str
    cpu_p95: float
    cpu_avg: float
    memory_p95: float | None  # None if memory agent not installed
    period_days: int


@dataclass
class RightsizingRecommendation:
    resource_id: str
    resource_type: str
    current_type: str
    recommended_type: str
    cpu_p95: float
    memory_p95: float | None
    reason: str
    estimated_savings_pct: float = 40.0  # rough estimate


def _get_cw_percentile(
    cw_client: Any,
    namespace: str,
    metric_name: str,
    dimensions: list[dict],
    start: datetime,
    end: datetime,
    percentile: int = 95,
) -> float | None:
    """Returns the Pn percentile value for a CloudWatch metric over the period."""
    try:
        response = cw_client.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric_name,
            Dimensions=dimensions,
            StartTime=start,
            EndTime=end,
            Period=3600,  # 1-hour granularity
            Statistics=["Average"],
            ExtendedStatistics=[f"p{percentile}"],
        )
    except Exception as exc:
        logger.warning("Failed to fetch %s/%s: %s", namespace, metric_name, exc)
        return None

    datapoints = response.get("Datapoints", [])
    if not datapoints:
        return None

    values = [dp.get(f"ExtendedStatistics", {}).get(f"p{percentile}", dp.get("Average", 0)) for dp in datapoints]
    return max(values) if values else None


def _analyze_ec2_instance(
    ec2_client: Any,
    cw_client: Any,
    instance: dict,
) -> UtilizationStats | None:
    instance_id = instance["InstanceId"]
    instance_type = instance["InstanceType"]
    region = ec2_client.meta.region_name

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)

    dims = [{"Name": "InstanceId", "Value": instance_id}]

    cpu_p95 = _get_cw_percentile(cw_client, "AWS/EC2", "CPUUtilization", dims, start, end, 95)
    cpu_avg_resp = cw_client.get_metric_statistics(
        Namespace="AWS/EC2",
        MetricName="CPUUtilization",
        Dimensions=dims,
        StartTime=start,
        EndTime=end,
        Period=86400,
        Statistics=["Average"],
    )
    avg_values = [dp["Average"] for dp in cpu_avg_resp.get("Datapoints", [])]
    cpu_avg = sum(avg_values) / len(avg_values) if avg_values else 0.0

    # Memory from CloudWatch agent (custom namespace)
    mem_p95 = _get_cw_percentile(
        cw_client, "CWAgent", "mem_used_percent", dims, start, end, 95
    )

    if cpu_p95 is None:
        logger.debug("No CPU data for %s, skipping", instance_id)
        return None

    return UtilizationStats(
        resource_id=instance_id,
        resource_type="EC2",
        instance_type=instance_type,
        region=region,
        cpu_p95=round(cpu_p95, 2),
        cpu_avg=round(cpu_avg, 2),
        memory_p95=round(mem_p95, 2) if mem_p95 is not None else None,
        period_days=LOOKBACK_DAYS,
    )


def _analyze_rds_instance(
    rds_client: Any,
    cw_client: Any,
    db: dict,
) -> UtilizationStats | None:
    db_id = db["DBInstanceIdentifier"]
    instance_type = db["DBInstanceClass"]
    region = rds_client.meta.region_name

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    dims = [{"Name": "DBInstanceIdentifier", "Value": db_id}]

    cpu_p95 = _get_cw_percentile(cw_client, "AWS/RDS", "CPUUtilization", dims, start, end, 95)
    if cpu_p95 is None:
        return None

    cpu_avg_resp = cw_client.get_metric_statistics(
        Namespace="AWS/RDS",
        MetricName="CPUUtilization",
        Dimensions=dims,
        StartTime=start,
        EndTime=end,
        Period=86400,
        Statistics=["Average"],
    )
    avg_values = [dp["Average"] for dp in cpu_avg_resp.get("Datapoints", [])]
    cpu_avg = sum(avg_values) / len(avg_values) if avg_values else 0.0

    return UtilizationStats(
        resource_id=db_id,
        resource_type="RDS",
        instance_type=instance_type,
        region=region,
        cpu_p95=round(cpu_p95, 2),
        cpu_avg=round(cpu_avg, 2),
        memory_p95=None,
        period_days=LOOKBACK_DAYS,
    )


def _generate_recommendation(stats: UtilizationStats) -> RightsizingRecommendation | None:
    downsize_map = EC2_DOWNSIZE_MAP if stats.resource_type == "EC2" else RDS_DOWNSIZE_MAP

    if stats.cpu_p95 >= CPU_DOWNSIZE_THRESHOLD_PCT:
        return None

    recommended = downsize_map.get(stats.instance_type)
    if not recommended:
        logger.debug("No downsize target for %s (%s)", stats.instance_type, stats.resource_id)
        return None

    reason_parts = [f"p95 CPU {stats.cpu_p95:.1f}% < {CPU_DOWNSIZE_THRESHOLD_PCT}%"]
    if stats.memory_p95 is not None and stats.memory_p95 < MEMORY_DOWNSIZE_THRESHOLD_PCT:
        reason_parts.append(f"p95 memory {stats.memory_p95:.1f}% < {MEMORY_DOWNSIZE_THRESHOLD_PCT}%")

    return RightsizingRecommendation(
        resource_id=stats.resource_id,
        resource_type=stats.resource_type,
        current_type=stats.instance_type,
        recommended_type=recommended,
        cpu_p95=stats.cpu_p95,
        memory_p95=stats.memory_p95,
        reason="; ".join(reason_parts),
    )


def analyze_rightsizing(
    regions: list[str] | None = None,
    session: boto3.Session | None = None,
) -> list[RightsizingRecommendation]:
    """Analyzes EC2 and RDS instances across regions and returns rightsizing recommendations."""
    if session is None:
        session = boto3.Session()
    if regions is None:
        regions = [session.region_name or "us-east-1"]

    recommendations: list[RightsizingRecommendation] = []

    for region in regions:
        ec2 = session.client("ec2", region_name=region)
        rds = session.client("rds", region_name=region)
        cw = session.client("cloudwatch", region_name=region)

        # EC2
        paginator = ec2.get_paginator("describe_instances")
        for page in paginator.paginate(Filters=[{"Name": "instance-state-name", "Values": ["running"]}]):
            for reservation in page["Reservations"]:
                for instance in reservation["Instances"]:
                    stats = _analyze_ec2_instance(ec2, cw, instance)
                    if stats:
                        rec = _generate_recommendation(stats)
                        if rec:
                            recommendations.append(rec)
                            logger.info("Recommend %s → %s for %s", rec.current_type, rec.recommended_type, rec.resource_id)

        # RDS
        rds_paginator = rds.get_paginator("describe_db_instances")
        for page in rds_paginator.paginate():
            for db in page["DBInstances"]:
                if db.get("DBInstanceStatus") != "available":
                    continue
                stats = _analyze_rds_instance(rds, cw, db)
                if stats:
                    rec = _generate_recommendation(stats)
                    if rec:
                        recommendations.append(rec)

    logger.info("Generated %d rightsizing recommendations", len(recommendations))
    return recommendations

# _r 20260703102914-76cd08aa
