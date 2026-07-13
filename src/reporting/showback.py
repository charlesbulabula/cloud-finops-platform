"""
Showback Report Generator — aggregates AWS costs by Team tag, generates a
Markdown table per team with month-over-month delta and top-3 services,
and posts to a Slack channel weekly.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import boto3
import requests

logger = logging.getLogger(__name__)

SLACK_WEBHOOK_ENV = "SLACK_SHOWBACK_WEBHOOK_URL"
SLACK_CHANNEL_ENV = "SLACK_SHOWBACK_CHANNEL"


@dataclass
class TeamCostSummary:
    team: str
    current_month_cost: float
    prior_month_cost: float
    mom_delta_pct: float
    top_services: list[tuple[str, float]]  # (service, cost)
    currency: str = "USD"


@dataclass
class ShowbackReport:
    generated_at: str
    current_period: str
    prior_period: str
    teams: list[TeamCostSummary] = field(default_factory=list)


def _month_boundaries(months_ago: int = 0) -> tuple[str, str]:
    """Returns (start, end) date strings for a calendar month."""
    today = date.today()
    first_of_current = today.replace(day=1)
    if months_ago == 0:
        start = first_of_current
        end = today
    else:
        # Go back N months
        month = first_of_current.month - months_ago
        year = first_of_current.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        start = date(year, month, 1)
        # Last day of that month
        next_month = date(year + (month // 12), (month % 12) + 1, 1)
        end = next_month - timedelta(days=1)
    return start.isoformat(), end.isoformat()


def _fetch_costs_by_team_and_service(
    ce_client: Any,
    start: str,
    end: str,
) -> dict[str, dict[str, float]]:
    """
    Returns {team: {service: cost}} for a given period.
    """
    response = ce_client.get_cost_and_usage(
        TimePeriod={"Start": start, "End": end},
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
        GroupBy=[
            {"Type": "TAG", "Key": "Team"},
            {"Type": "DIMENSION", "Key": "SERVICE"},
        ],
    )

    result: dict[str, dict[str, float]] = {}
    for time_result in response.get("ResultsByTime", []):
        for group in time_result.get("Groups", []):
            keys = group.get("Keys", [])
            if len(keys) < 2:
                continue
            team_raw = keys[0]
            service = keys[1]
            team = team_raw.replace("Team$", "") or "untagged"
            amount = float(group["Metrics"]["UnblendedCost"]["Amount"])
            if team not in result:
                result[team] = {}
            result[team][service] = result[team].get(service, 0.0) + amount

    return result


def _compute_mom_delta(current: float, prior: float) -> float:
    if prior == 0:
        return 100.0 if current > 0 else 0.0
    return round((current - prior) / prior * 100, 1)


def _build_report(
    current_costs: dict[str, dict[str, float]],
    prior_costs: dict[str, dict[str, float]],
    current_period: str,
    prior_period: str,
) -> ShowbackReport:
    all_teams = set(current_costs.keys()) | set(prior_costs.keys())
    summaries: list[TeamCostSummary] = []

    for team in sorted(all_teams):
        cur_by_service = current_costs.get(team, {})
        prior_by_service = prior_costs.get(team, {})

        current_total = sum(cur_by_service.values())
        prior_total = sum(prior_by_service.values())
        mom_delta = _compute_mom_delta(current_total, prior_total)

        top_services = sorted(cur_by_service.items(), key=lambda x: x[1], reverse=True)[:3]

        summaries.append(
            TeamCostSummary(
                team=team,
                current_month_cost=round(current_total, 2),
                prior_month_cost=round(prior_total, 2),
                mom_delta_pct=mom_delta,
                top_services=[(s, round(c, 2)) for s, c in top_services],
            )
        )

    return ShowbackReport(
        generated_at=date.today().isoformat(),
        current_period=current_period,
        prior_period=prior_period,
        teams=summaries,
    )


def render_markdown(report: ShowbackReport) -> str:
    lines = [
        f"# AWS Showback Report",
        f"**Generated:** {report.generated_at}  ",
        f"**Current Period:** {report.current_period}  ",
        f"**Prior Period:** {report.prior_period}",
        "",
        "| Team | Current Month | Prior Month | MoM Delta | Top Services |",
        "|------|--------------|-------------|-----------|--------------|",
    ]

    for team in report.teams:
        delta_str = f"+{team.mom_delta_pct}%" if team.mom_delta_pct >= 0 else f"{team.mom_delta_pct}%"
        top = ", ".join(f"{s} (${c:.0f})" for s, c in team.top_services)
        lines.append(
            f"| {team.team} | ${team.current_month_cost:,.2f} | ${team.prior_month_cost:,.2f} | {delta_str} | {top} |"
        )

    return "\n".join(lines)


def _send_to_slack(markdown: str, webhook_url: str, channel: str | None = None) -> None:
    payload: dict = {
        "text": "AWS Weekly Showback Report",
        "blocks": [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": markdown[:3000]},
            }
        ],
    }
    if channel:
        payload["channel"] = channel

    resp = requests.post(webhook_url, json=payload, timeout=10)
    if resp.status_code != 200:
        logger.error("Slack post failed: %s %s", resp.status_code, resp.text)
    else:
        logger.info("Showback report sent to Slack")


def generate_and_send_showback(
    session: boto3.Session | None = None,
    slack_webhook: str | None = None,
    slack_channel: str | None = None,
    dry_run: bool = False,
) -> ShowbackReport:
    """Fetches cost data, builds showback report, and sends to Slack."""
    if session is None:
        session = boto3.Session()
    if slack_webhook is None:
        slack_webhook = os.environ.get(SLACK_WEBHOOK_ENV)
    if slack_channel is None:
        slack_channel = os.environ.get(SLACK_CHANNEL_ENV)

    ce = session.client("ce", region_name="us-east-1")

    current_start, current_end = _month_boundaries(0)
    prior_start, prior_end = _month_boundaries(1)

    logger.info("Fetching current period costs: %s to %s", current_start, current_end)
    current_costs = _fetch_costs_by_team_and_service(ce, current_start, current_end)

    logger.info("Fetching prior period costs: %s to %s", prior_start, prior_end)
    prior_costs = _fetch_costs_by_team_and_service(ce, prior_start, prior_end)

    report = _build_report(
        current_costs, prior_costs,
        f"{current_start} to {current_end}",
        f"{prior_start} to {prior_end}",
    )

    markdown = render_markdown(report)
    logger.debug("Generated report:\n%s", markdown)

    if slack_webhook and not dry_run:
        _send_to_slack(markdown, slack_webhook, slack_channel)
    elif dry_run:
        logger.info("[DRY RUN] Would send showback report to Slack:\n%s", markdown)

    return report

# _r 20260713111214-bd3da2b3
