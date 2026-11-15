"""Scheduled events that move prices: an NYSE trading calendar and company earnings dates.

Earnings come from Yahoo Finance (``yfinance``), which lists past releases and the next one
with their time of day. :func:`event_sessions` turns release times into the trading session
whose close-to-close return carries the move: a release before the open moves that day's
session, one during or after the session moves the next.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay

logger = logging.getLogger(__name__)

__all__ = [
    "NYSEHolidayCalendar",
    "NYSE_DAY",
    "trading_days_after",
    "event_sessions",
    "fetch_earnings_dates",
]

NY = "America/New_York"


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """Regular NYSE holidays (one-off closures such as days of mourning are not included)."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


#: One NYSE trading day, for date arithmetic (``ts + NYSE_DAY``).
NYSE_DAY = CustomBusinessDay(calendar=NYSEHolidayCalendar())


def trading_days_after(day: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """The ``n`` NYSE sessions after ``day`` (``day`` itself excluded)."""
    start = pd.Timestamp(day).normalize() + NYSE_DAY
    return pd.date_range(start, periods=n, freq=NYSE_DAY)


def event_sessions(times: Iterable[pd.Timestamp], *, cutoff_hour: int = 12) -> pd.DatetimeIndex:
    """Map release times to the session whose close-to-close return carries the move.

    Args:
        times: Release timestamps. Naive ones are read as New York time.
        cutoff_hour: Releases at or after this New York hour move the next session
            (after-the-close reports at 16:00 or later); earlier ones move that day's
            session (pre-market reports), or the next session if that day is not one.

    Returns:
        Sorted unique session dates (midnight, naive).
    """
    out = set()
    for t in times:
        ts = pd.Timestamp(t)
        ts = ts.tz_localize(NY) if ts.tzinfo is None else ts.tz_convert(NY)
        day = pd.Timestamp(ts.date())
        if ts.hour >= cutoff_hour or not NYSE_DAY.is_on_offset(day):
            day = day + NYSE_DAY
        out.add(day)
    return pd.DatetimeIndex(sorted(out))


def fetch_earnings_dates(symbol: str, limit: int = 100) -> pd.DatetimeIndex:
    """Past and upcoming earnings release times from Yahoo Finance (New York time).

    Upcoming dates can be estimates until the company confirms them. Returns an empty
    index for symbols without earnings (ETFs) or when Yahoo has nothing.
    """
    import yfinance as yf

    try:
        df = yf.Ticker(symbol).get_earnings_dates(limit=limit)
    except Exception as exc:  # noqa: BLE001 - no earnings is a valid answer for ETFs
        logger.info("no earnings dates for %s: %s", symbol, exc)
        return pd.DatetimeIndex([], tz=NY)
    if df is None or df.empty:
        return pd.DatetimeIndex([], tz=NY)
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize(NY) if idx.tz is None else idx.tz_convert(NY)
    return idx.sort_values().unique()
