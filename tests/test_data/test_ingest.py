"""Tests for data ingestion module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from augury.data.ingest import DataIngestor, DataSource, YFinanceSource


class ConcreteSource(DataSource):
    """Minimal concrete DataSource for testing."""

    @property
    def name(self) -> str:
        return "test_source"

    def fetch(self, symbol, start, end, interval="1d"):
        return pd.DataFrame()


def test_data_source_is_abstract() -> None:
    """DataSource cannot be instantiated directly (abstract class)."""
    with pytest.raises(TypeError):
        DataSource()  # type: ignore[abstract]


def test_concrete_source_can_be_instantiated() -> None:
    """A concrete DataSource subclass can be instantiated."""
    src = ConcreteSource()
    assert src.name == "test_source"


def test_yfinance_source_name() -> None:
    """YFinanceSource.name returns 'yahoo_finance'."""
    src = YFinanceSource()
    assert src.name == "yahoo_finance"


def test_data_ingestor_raises_when_all_sources_fail() -> None:
    """DataIngestor.fetch raises RuntimeError if every source fails."""

    class FailingSource(DataSource):
        @property
        def name(self) -> str:
            return "failing"

        def fetch(self, symbol, start, end, interval="1d"):
            raise ConnectionError("Network unavailable")

    ingestor = DataIngestor(sources=[FailingSource()])
    with pytest.raises(RuntimeError, match="All data sources failed"):
        ingestor.fetch("SPY", "2022-01-01", "2023-01-01")


def test_data_ingestor_fetch_multiple_handles_partial_failures() -> None:
    """fetch_multiple returns successful symbols even if some fail."""
    n = 10
    good_df = pd.DataFrame(
        {"close": [100.0] * n},
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
    )

    class PartialSource(DataSource):
        @property
        def name(self) -> str:
            return "partial"

        def fetch(self, symbol, start, end, interval="1d"):
            if symbol == "GOOD":
                return good_df
            raise ValueError(f"No data for {symbol}")

    ingestor = DataIngestor(sources=[PartialSource()])
    results = ingestor.fetch_multiple(["GOOD", "BAD"], start="2022-01-01", end="2023-01-01")
    assert "GOOD" in results
    assert "BAD" not in results
    assert len(results["GOOD"]) == n


def test_data_ingestor_uses_first_successful_source() -> None:
    """DataIngestor tries sources in order and returns the first success."""
    calls: list[str] = []
    good_df = pd.DataFrame(
        {"close": [100.0]},
        index=pd.date_range("2022-01-01", periods=1, freq="B"),
    )

    class FirstSource(DataSource):
        @property
        def name(self) -> str:
            return "first"

        def fetch(self, symbol, start, end, interval="1d"):
            calls.append("first")
            raise RuntimeError("First failed")

    class SecondSource(DataSource):
        @property
        def name(self) -> str:
            return "second"

        def fetch(self, symbol, start, end, interval="1d"):
            calls.append("second")
            return good_df

    ingestor = DataIngestor(sources=[FirstSource(), SecondSource()])
    result = ingestor.fetch("SPY", "2022-01-01", "2022-01-02")
    assert calls == ["first", "second"]
    assert not result.empty


# ── YFinanceSource.fetch with mocked yfinance ─────────────────────────────────


def _make_yf_dataframe(tz: str | None = "America/New_York") -> pd.DataFrame:
    """Build a DataFrame mimicking yfinance Ticker.history() output."""
    dates = pd.date_range("2023-01-03", periods=5, freq="B")
    if tz:
        dates = dates.tz_localize(tz)
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [1e6, 2e6, 3e6, 4e6, 5e6],
            "Dividends": [0, 0, 0, 0, 0],
            "Stock Splits": [0, 0, 0, 0, 0],
        },
        index=dates,
    )


@patch("augury.data.ingest.yf", create=True)
def test_yfinance_fetch_returns_ohlcv(mock_yf: MagicMock) -> None:
    """YFinanceSource.fetch returns normalized OHLCV DataFrame."""
    # We patch at the import location inside the fetch method
    raw = _make_yf_dataframe()
    with patch.dict("sys.modules", {"yfinance": MagicMock()}) as _:
        import sys

        fake_yf = sys.modules["yfinance"]
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = raw
        fake_yf.Ticker.return_value = mock_ticker

        src = YFinanceSource()
        result = src.fetch("SPY", "2023-01-01", "2023-01-10")

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"open", "high", "low", "close", "volume"}
    assert len(result) == 5


@patch.dict("sys.modules", {"yfinance": MagicMock()})
def test_yfinance_fetch_removes_timezone() -> None:
    """YFinanceSource._normalize strips timezone info from index."""
    import sys

    fake_yf = sys.modules["yfinance"]
    raw = _make_yf_dataframe(tz="America/New_York")
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = raw
    fake_yf.Ticker.return_value = mock_ticker

    src = YFinanceSource()
    result = src.fetch("AAPL", "2023-01-01", "2023-01-10")
    assert result.index.tz is None


@patch.dict("sys.modules", {"yfinance": MagicMock()})
def test_yfinance_fetch_no_tz_still_works() -> None:
    """Works fine when yfinance returns data without timezone."""
    import sys

    fake_yf = sys.modules["yfinance"]
    raw = _make_yf_dataframe(tz=None)
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = raw
    fake_yf.Ticker.return_value = mock_ticker

    src = YFinanceSource()
    result = src.fetch("MSFT", "2023-01-01", "2023-01-10")
    assert result.index.tz is None
    assert len(result) == 5


@patch.dict("sys.modules", {"yfinance": MagicMock()})
def test_yfinance_fetch_empty_raises_value_error() -> None:
    """Raises ValueError when yfinance returns empty DataFrame."""
    import sys

    fake_yf = sys.modules["yfinance"]
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    fake_yf.Ticker.return_value = mock_ticker

    src = YFinanceSource()
    with pytest.raises(ValueError, match="No data returned"):
        src.fetch("INVALID", "2023-01-01", "2023-01-10")


@patch.dict("sys.modules", {"yfinance": MagicMock()})
def test_yfinance_fetch_lowercases_columns() -> None:
    """Column names are lowercased during normalization."""
    import sys

    fake_yf = sys.modules["yfinance"]
    raw = _make_yf_dataframe(tz=None)
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = raw
    fake_yf.Ticker.return_value = mock_ticker

    src = YFinanceSource()
    result = src.fetch("SPY", "2023-01-01", "2023-01-10")
    # Extra columns like "dividends" and "stock splits" should be dropped
    assert "dividends" not in result.columns
    assert "stock splits" not in result.columns


@patch.dict("sys.modules", {"yfinance": MagicMock()})
def test_yfinance_auto_adjust_passed() -> None:
    """auto_adjust parameter is forwarded to yfinance."""
    import sys

    fake_yf = sys.modules["yfinance"]
    raw = _make_yf_dataframe(tz=None)
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = raw
    fake_yf.Ticker.return_value = mock_ticker

    src = YFinanceSource(auto_adjust=False)
    src.fetch("SPY", "2023-01-01", "2023-01-10")
    call_kwargs = mock_ticker.history.call_args[1]
    assert call_kwargs["auto_adjust"] is False


def test_yfinance_normalize_directly() -> None:
    """_normalize works on a raw DataFrame without yfinance import."""
    raw = _make_yf_dataframe(tz="UTC")
    src = YFinanceSource()
    result = src._normalize(raw)
    assert result.index.tz is None
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert len(result) == 5
    # Values should match the original
    np.testing.assert_allclose(result["close"].values, [100.5, 101.5, 102.5, 103.5, 104.5])


# ── DataIngestor default constructor ──────────────────────────────────────────


def test_data_ingestor_default_sources() -> None:
    """Default DataIngestor creates a YFinanceSource."""
    ingestor = DataIngestor()
    assert len(ingestor._sources) == 1
    assert isinstance(ingestor._sources[0], YFinanceSource)
