"""Publish augury's daily risk forecasts into market-compass.

Run by the hub's "augury" job (market-compass/registry.py) with this project's venv. The hub
puts its folder on PYTHONPATH (for ``core.store``) and sets HUB_DATA_DIR; ``--demo`` uses
synthetic prices and makes no network calls.

Writes data/augury/:
  forecast   one row per ticker x horizon (1, 5, 21 days): VaR / ES, 90% price range,
             forecast vol, P(up)
  bands      daily price quantiles for the next 21 trading days (fan chart)
  prices     last year of closes (chart context)
  track      every out-of-sample forecast since TRACK_START with its outcome
  summary    as_of, models, headline track-record scores per ticker, errors

The track record is extended each night: only the last few weeks are recomputed (to fill
in the 5- and 21-day outcomes that closed), so the first run takes several minutes and
later runs about a minute.

    python scripts/publish_to_hub.py [--demo] [--only SPY,QQQ] [--rebuild]
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from augury import __version__
from augury.forecasters.price import default_forecaster
from augury.monitoring import (
    IV_VIEWS,
    last_final_session,
    live_forecast,
    monitor_model,
    score_track,
    track_models,
    track_record,
)

try:
    from core import store  # market-compass, on PYTHONPATH when run by the hub
except ImportError:  # pragma: no cover - only when run by hand without the hub
    sys.exit("Run this through market-compass (python run_jobs.py --only augury), or put the hub "
             "folder on PYTHONPATH and set HUB_DATA_DIR.")

MODULE = "augury"
GROUPS: dict[str, list[str]] = {
    "Index ETFs": ["SPY", "QQQ", "IWM", "DIA"],
    "Sectors": ["XLF", "XLE", "XLK", "XLV", "XLI", "XLU", "SMH"],
    "Rates & credit": ["TLT", "IEF", "HYG", "LQD"],
    "International": ["EEM", "EFA"],
    "Commodities": ["GLD", "SLV", "USO"],
    "Single stocks": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM"],
}
TRACK_START = "2016-01-01"
HISTORY_START = "2013-01-01"  # two years of fitting window before TRACK_START

warnings.filterwarnings("ignore")


def log(msg: str) -> None:
    print(f"[augury {datetime.now():%H:%M:%S}] {msg}", flush=True)


# ----------------------------------------------------------------- data
def fetch_real(symbols: list[str]) -> dict[str, pd.Series]:
    from augury.data.ingest import DataIngestor

    ingestor, cutoff, out = DataIngestor(), pd.Timestamp(last_final_session()), {}
    for s in symbols:
        try:
            df = ingestor.fetch(s, HISTORY_START, date.today() + timedelta(days=1))
            close = df["close"].dropna()
            close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
            out[s] = close[close.index <= cutoff]
        except Exception as exc:  # noqa: BLE001 - one bad symbol shouldn't stop the run
            log(f"fetch failed for {s}: {exc}")
    return out


def fetch_demo(symbols: list[str]) -> dict[str, pd.Series]:
    """Synthetic GARCH-like prices and a matching vol index; no network."""
    days = pd.bdate_range("2019-01-01", "2026-03-31")
    out = {}
    for k, s in enumerate(x for x in symbols if not x.startswith("^")):
        rng = np.random.default_rng(100 + k)
        h, var, rets = days.size, np.empty(days.size), np.empty(days.size)
        var[0] = 1e-4
        for t in range(h):
            if t:
                var[t] = 2e-6 + 0.08 * rets[t - 1] ** 2 + 0.9 * var[t - 1]
            rets[t] = np.sqrt(var[t]) * rng.standard_t(6) / np.sqrt(1.5)
        out[s] = pd.Series(100 * np.exp(np.cumsum(rets)), index=days)
        out[f"^{s}VOL"] = pd.Series(np.sqrt(var * 252) * 100 * 1.1, index=days)
    return out


# ----------------------------------------------------------------- run
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic data, no network")
    ap.add_argument("--only", help="comma-separated tickers (default: all)")
    ap.add_argument("--rebuild", action="store_true", help="recompute the whole track record")
    args = ap.parse_args()

    groups = {"Demo": ["SPY", "TLT", "GLD", "AAPL"]} if args.demo else GROUPS
    group_of = {t: g for g, ts in groups.items() for t in ts}
    tickers = [t for t in group_of if not args.only or t in args.only.upper().split(",")]
    iv_symbol = {t: (f"^{t}VOL" if args.demo else IV_VIEWS[t][0]) for t in tickers if t in IV_VIEWS}
    track_start = "2023-01-01" if args.demo else TRACK_START

    t0 = time.time()
    data = (fetch_demo if args.demo else fetch_real)(tickers + sorted(set(iv_symbol.values())))
    log(f"fetched {len(data)} series in {time.time() - t0:.0f}s")

    old_summary = store.load_json(MODULE, "summary") or {}
    previous = None
    if not args.demo and not args.rebuild and not old_summary.get("demo"):
        previous = store.load_frame(MODULE, "track")

    forecasts, bands, prices, tracks, scores, models, errors = [], [], [], [], {}, {}, {}
    for t in tickers:
        try:
            close = data.get(t)
            if close is None or len(close) < 600:
                raise ValueError(f"not enough price history ({0 if close is None else len(close)} days)")
            iv = data.get(iv_symbol.get(t, ""))
            iv = iv / 100.0 if iv is not None else None
            model = monitor_model(t, iv)
            risk, band = live_forecast(close, model)
            gjr_risk, _ = live_forecast(close, default_forecaster(), n_samples=5000)
            r = np.log(close).diff().dropna()
            risk = risk.assign(ticker=t, group=group_of[t], gjr_vol_ann=gjr_risk["vol_ann"].to_numpy(),
                               realised_vol_21d=float(r.iloc[-21:].std() * np.sqrt(252)),
                               iv=float(iv.dropna().iloc[-1]) if iv is not None and t in IV_VIEWS else np.nan,
                               iv_symbol=iv_symbol.get(t))
            forecasts.append(risk)
            bands.append(band.reset_index().assign(ticker=t, day=np.arange(1, len(band) + 1)))
            prices.append(pd.DataFrame({"ticker": t, "date": close.index[-260:],
                                        "close": close.to_numpy()[-260:]}))

            prev = None if previous is None else previous[previous["ticker"] == t].drop(columns="ticker")
            rec = track_record(close, track_models(t, iv), start=track_start, previous=prev)
            tracks.append(rec.assign(ticker=t))
            base = "gjr" if (rec["model"] == "gjr").any() else None
            scores[t] = score_track(rec, "augury", baseline=base)
            models[t] = model.name
            log(f"{t}: {model.name}, 1d VaR99 {risk['var_99'].iloc[0]:+.2%}, "
                f"{len(rec) // max(rec['model'].nunique() * 3, 1)} days tracked")
        except Exception as exc:  # noqa: BLE001
            errors[t] = f"{type(exc).__name__}: {exc}"
            log(f"{t} failed: {errors[t]}")

    if not forecasts:
        log("nothing published")
        return 1

    def save(name: str, frames: list[pd.DataFrame]) -> pd.DataFrame:
        df = pd.concat(frames, ignore_index=True)
        old = store.load_frame(MODULE, name) if args.only else None
        if old is not None:  # --only: keep the tickers this run didn't touch
            df = pd.concat([old[~old["ticker"].isin(df["ticker"].unique())], df], ignore_index=True)
        store.save_frame(MODULE, name, df)
        return df

    fc = save("forecast", forecasts)
    save("bands", bands)
    save("prices", prices)
    save("track", tracks)
    if args.only:
        scores = {**old_summary.get("scores", {}), **scores}
        models = {**old_summary.get("models", {}), **models}
    store.save_json(MODULE, "summary", {
        "as_of": fc["as_of"].max(),
        "published": datetime.now().isoformat(timespec="seconds"),
        "demo": args.demo,
        "augury_version": __version__,
        "groups": groups,
        "models": models,
        "iv_views": iv_symbol,
        "track_start": track_start,
        "scores": scores,
        "errors": errors,
        "seconds": round(time.time() - t0),
    })
    log(f"published {len(forecasts)} tickers in {time.time() - t0:.0f}s"
        + (f"; {len(errors)} failed: {', '.join(errors)}" if errors else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
