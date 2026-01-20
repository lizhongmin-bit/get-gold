"""Trading calendar and time utilities."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Iterable, Tuple


TRADING_SESSIONS = ((time(9, 30), time(11, 30)), (time(13, 0), time(15, 0)))
TAIL_WINDOW = (time(14, 30), time(15, 0))


def is_trading_time(ts: datetime) -> bool:
    current = ts.time()
    return any(start <= current <= end for start, end in TRADING_SESSIONS)


def tail_window_times() -> Tuple[time, time]:
    return TAIL_WINDOW


def most_recent_trading_date(current: datetime | None = None) -> date:
    now = current or datetime.now()
    candidate = now.date()
    if now.weekday() >= 5 or not is_trading_time(now):
        candidate = _previous_weekday(candidate)
    return candidate


def _previous_weekday(current: date) -> date:
    candidate = current - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def filter_tail_window(index: Iterable[datetime]) -> list[datetime]:
    start, end = tail_window_times()
    return [ts for ts in index if start <= ts.time() <= end]
