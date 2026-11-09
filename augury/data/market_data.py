"""Fetch and cache market-implied data for predictive signals.

Provides time-aligned arrays of:
  - VIX (30-day implied vol)
  - VIX3M (3-month implied vol)
  - SKEW (CBOE tail-risk index)
  - MOVE (bond vol index)
  - HYG (high-yield corporate bond ETF)
  - IEF (7-10yr Treasury ETF)

All data is fetched once and cached, then sliced by date for each
walk-forward step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["MarketDataFetcher", "MarketSnapshot"]

# Tickers we fetch for market-implied signals
_MARKET_TICKERS: dict[str, str] = {
    "vix": "^VIX",
    "vix3m": "^VIX3M",
    "skew": "^SKEW",
    "move": "^MOVE",
    "hyg": "HYG",
    "ief": "IEF",
}


@dataclass
class MarketSnapshot:
    """Aligned market data arrays for a given date window.

    All arrays have the same length and cover the same trading days.
    Values are np.nan where a particular series had no data on that day.
    """

    dates: np.ndarray  # datetime64 array
    vix: np.ndarray | None = None
    vix3m: np.ndarray | None = None
    skew: np.ndarray | None = None
    move: np.ndarray | None = None
    hyg_close: np.ndarray | None = None
    ief_close: np.ndarray | None = None

    @property
    def n_days(self) -> int:
        return len(self.dates)

    def has_vix_term_structure(self) -> bool:
        return self.vix is not None and self.vix3m is not None

    def has_credit_spread(self) -> bool:
        return self.hyg_close is not None and self.ief_close is not None


class MarketDataFetcher:
    """Fetch and cache market-implied data from yfinance.

    Fetches all external tickers once at construction, then provides
    fast date-aligned slicing for walk-forward loops.

    Args:
        start: Start date for data fetch (should pre-date your backtest
            by at least the lookback window).
        end: End date for data fetch.
        tickers: Override which tickers to fetch.  Keys must be from
            {'vix', 'vix3m', 'skew', 'move', 'hyg', 'ief'}.
    """

    def __init__(
        self,
        start: str = "2005-01-01",
        end: str = "2026-12-31",
        tickers: dict[str, str] | None = None,
    ) -> None:
        self._tickers = dict(tickers or _MARKET_TICKERS)
        self._data: dict[str, pd.Series] = {}
        self._fetch(start, end)

    def _fetch(self, start: str, end: str) -> None:
        """Fetch all tickers and store as date-indexed Series."""
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed — market data unavailable")
            return

        for key, symbol in self._tickers.items():
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start, end=end, auto_adjust=True)
                if df.empty:
                    logger.warning(f"No data for {symbol} ({key})")
                    continue
                s = df["Close"].copy()
                s.index = pd.to_datetime(s.index)
                if s.index.tz is not None:
                    s.index = s.index.tz_localize(None)
                self._data[key] = s
                logger.info(f"Fetched {symbol}: {len(s)} days")
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol} ({key}): {e}")

    @property
    def available_series(self) -> list[str]:
        """Names of successfully fetched series."""
        return list(self._data.keys())

    def slice(
        self,
        start_date: Any,
        end_date: Any,
    ) -> MarketSnapshot:
        """Get market data aligned to a date range.

        Args:
            start_date: First date (inclusive).
            end_date: Last date (inclusive).

        Returns:
            MarketSnapshot with arrays covering the requested range.
        """
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        # Build a common date index from all available series
        all_dates: set[pd.Timestamp] = set()
        for s in self._data.values():
            mask = (s.index >= start_ts) & (s.index <= end_ts)
            all_dates.update(s.index[mask])

        if not all_dates:
            return MarketSnapshot(dates=np.array([], dtype="datetime64[ns]"))

        dates = np.array(sorted(all_dates))

        def _align(key: str) -> np.ndarray | None:
            if key not in self._data:
                return None
            s = self._data[key]
            # Reindex to common dates, forward-fill gaps (weekends etc.)
            aligned = s.reindex(pd.DatetimeIndex(dates), method="ffill")
            return aligned.to_numpy(dtype=float)

        return MarketSnapshot(
            dates=dates,
            vix=_align("vix"),
            vix3m=_align("vix3m"),
            skew=_align("skew"),
            move=_align("move"),
            hyg_close=_align("hyg"),
            ief_close=_align("ief"),
        )

    def slice_by_index(
        self,
        equity_index: pd.DatetimeIndex,
    ) -> MarketSnapshot:
        """Get market data aligned to an equity DataFrame's index.

        This is convenient for walk-forward: pass the index of your
        equity OHLCV slice and get back market data on matching dates.
        """
        if len(equity_index) == 0:
            return MarketSnapshot(dates=np.array([], dtype="datetime64[ns]"))

        idx = equity_index
        if idx.tz is not None:
            idx = idx.tz_localize(None)

        return self.slice(idx[0], idx[-1])
