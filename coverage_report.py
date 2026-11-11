"""
Coverage report: which tournament questions opened recently, how long each
one stayed open for forecasting, and whether this bot got a forecast in.

Read-only. It makes no forecasts and no model calls, so it costs nothing to
run. It prints only ids, times and yes/no flags - never question wording - so
reading its output does not put a human in the loop on open questions.

Run from GitHub: Actions -> "Coverage report" -> Run workflow.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
from datetime import datetime, timedelta, timezone

import dotenv
from forecasting_tools import ApiFilter, MetaculusClient, MetaculusQuestion

dotenv.load_dotenv()


def _hours(delta: timedelta) -> float:
    return delta.total_seconds() / 3600


def _fmt(moment: datetime | None) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M") if moment else "unknown         "


async def _recent_questions(
    client: MetaculusClient, tournament: str | int, since: datetime
) -> list[MetaculusQuestion]:
    api_filter = ApiFilter(
        allowed_tournaments=[tournament],
        open_time_gt=since,
        group_question_mode="unpack_subquestions",
    )
    return await client.get_questions_matching_filter(
        api_filter, num_questions=500, error_if_question_target_missed=False
    )


def build_report(
    name: str, questions: list[MetaculusQuestion], now: datetime, live_since: datetime
) -> tuple[str, dict[str, int]]:
    lines = [f"=== {name}: {len(questions)} question(s) opened in the period ==="]
    lines.append("post id | opened (UTC)     | closes (UTC)     | open for | status   | forecast by bot")
    windows: list[float] = []
    counts = {"forecast": 0, "missed": 0, "still_open_unforecast": 0, "before_live": 0}
    for question in sorted(questions, key=lambda q: q.open_time or now, reverse=True):
        opened, closes = question.open_time, question.close_time
        window = _hours(closes - opened) if opened and closes else None
        if window is not None:
            windows.append(window)
        is_closed = closes is not None and closes <= now
        forecasted = bool(question.already_forecasted)
        if forecasted:
            verdict = "yes"
            counts["forecast"] += 1
        elif closes is not None and closes <= live_since:
            verdict = "no (closed before bot went live)"
            counts["before_live"] += 1
        elif is_closed:
            verdict = "NO - MISSED"
            counts["missed"] += 1
        else:
            verdict = "not yet (still open)"
            counts["still_open_unforecast"] += 1
        window_text = f"{window:7.1f}h" if window is not None else "      ?"
        status = question.state.value if question.state else ("closed" if is_closed else "open")
        lines.append(
            f"{question.id_of_post:>7} | {_fmt(opened)} | {_fmt(closes)} | {window_text} | {status:<8} | {verdict}"
        )
    if windows:
        lines.append(
            f"Open-for window in hours: shortest {min(windows):.1f}, median {statistics.median(windows):.1f}, "
            f"longest {max(windows):.1f}; {sum(1 for w in windows if w <= 3)} of {len(windows)} were open 3 hours or less."
        )
    lines.append(
        f"Since the bot went live: forecast {counts['forecast']}, MISSED {counts['missed']}, "
        f"still open and not yet forecast {counts['still_open_unforecast']}."
    )
    return "\n".join(lines), counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Report recent question coverage")
    parser.add_argument("--days", type=float, default=7, help="How many days back to look")
    parser.add_argument(
        "--live-since",
        default="2026-11-10T06:44:00+00:00",
        help="When the bot started running (ISO time); earlier questions are not counted as missed",
    )
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=args.days)
    live_since = datetime.fromisoformat(args.live_since)
    client = MetaculusClient()
    print(f"Coverage report at {_fmt(now)} UTC, looking back {args.days:g} days. Bot live since {_fmt(live_since)} UTC.\n")
    for name, tournament in (
        ("Seasonal tournament", client.CURRENT_AI_COMPETITION_ID),
        ("MiniBench", client.CURRENT_MINIBENCH_ID),
    ):
        questions = asyncio.run(_recent_questions(client, tournament, since))
        report, _ = build_report(name, questions, now, live_since)
        print(report + "\n")


if __name__ == "__main__":
    main()
