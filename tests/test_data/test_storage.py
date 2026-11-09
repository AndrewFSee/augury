"""Tests for Parquet-based market data storage."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from augury.data.storage import ParquetStorage, Storage


@pytest.fixture()
def tmp_dir():
    """Create a temp directory that works on Windows."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def storage(tmp_dir) -> ParquetStorage:
    """Create a ParquetStorage rooted in a temp directory."""
    return ParquetStorage(base_dir=tmp_dir / "market")


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Small OHLCV DataFrame for testing."""
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=idx,
    )


# ── ParquetStorage basics ────────────────────────────────────────────────────


class TestParquetStorage:
    def test_is_storage_subclass(self, storage) -> None:
        assert isinstance(storage, Storage)

    def test_save_creates_file(self, storage, sample_df, tmp_dir) -> None:
        storage.save("AAPL", sample_df)
        expected_path = tmp_dir / "market" / "AAPL.parquet"
        assert expected_path.exists()

    def test_load_round_trip(self, storage, sample_df) -> None:
        storage.save("AAPL", sample_df)
        loaded = storage.load("AAPL")
        pd.testing.assert_frame_equal(loaded, sample_df, check_freq=False)

    def test_load_missing_raises_key_error(self, storage) -> None:
        with pytest.raises(KeyError, match="MISSING"):
            storage.load("MISSING")

    def test_list_symbols_empty(self, storage) -> None:
        assert storage.list_symbols() == []

    def test_list_symbols_after_save(self, storage, sample_df) -> None:
        storage.save("aapl", sample_df)
        storage.save("msft", sample_df)
        symbols = storage.list_symbols()
        assert symbols == ["AAPL", "MSFT"]

    def test_symbol_case_insensitive(self, storage, sample_df) -> None:
        """Saving with lowercase, loading with uppercase should work."""
        storage.save("spy", sample_df)
        loaded = storage.load("SPY")
        pd.testing.assert_frame_equal(loaded, sample_df, check_freq=False)

    def test_overwrite_existing(self, storage, sample_df) -> None:
        storage.save("AAPL", sample_df)
        new_df = sample_df.copy()
        new_df["close"] = 999.0
        storage.save("AAPL", new_df)
        loaded = storage.load("AAPL")
        assert (loaded["close"] == 999.0).all()

    def test_delete_removes_file(self, storage, sample_df) -> None:
        storage.save("AAPL", sample_df)
        storage.delete("AAPL")
        assert storage.list_symbols() == []

    def test_delete_nonexistent_no_error(self, storage) -> None:
        """Deleting a symbol that doesn't exist should not raise."""
        storage.delete("NOPE")  # just logs a warning

    def test_base_dir_created_on_init(self, tmp_dir) -> None:
        new_dir = tmp_dir / "deep" / "nested" / "dir"
        assert not new_dir.exists()
        ParquetStorage(base_dir=new_dir)
        assert new_dir.exists()
