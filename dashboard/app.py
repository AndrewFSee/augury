"""Augury risk monitor: today's forecast distribution and its out-of-sample track record.

    streamlit run dashboard/app.py

Prices come from Yahoo Finance (cached for a few hours). The same numbers are published
nightly to the market-compass hub by scripts/publish_to_hub.py; this app adds any ticker,
longer horizons and your own drift / volatility views.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plots import fan_chart, plot_pit_histogram, return_histogram, track_chart  # noqa: E402

from augury.forecasters.price import PriceForecast, forecast_prices  # noqa: E402
from augury.forecasters.views import ImpliedVolForecaster  # noqa: E402
from augury.monitoring import (  # noqa: E402
    IV_VIEWS,
    last_final_session,
    monitor_model,
    risk_table,
    score_track,
    track_models,
    track_record,
)

TRACK_START = "2016-01-01"
WINDOWS = {"1y": 252, "3y": 756, "All": None}
INFO = {"delta_color": "off", "delta_arrow": "off", "border": True}  # metric deltas that are context, not change

st.set_page_config(page_title="Augury risk monitor", page_icon=":material/query_stats:", layout="wide")


# ----------------------------------------------------------------- data (cached)
@st.cache_data(ttl="6h", show_spinner="Downloading prices…")
def closes(symbol: str, through: date) -> pd.Series:
    from augury.data.ingest import DataIngestor

    df = DataIngestor().fetch(symbol, "2013-01-01", date.today() + timedelta(days=1))
    s = df["close"].dropna()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s[s.index <= pd.Timestamp(through)]


def implied_vol(ticker: str, through: date) -> pd.Series | None:
    view = IV_VIEWS.get(ticker)
    return closes(view[0], through) / 100.0 if view else None


@st.cache_data(ttl="6h", show_spinner="Fitting the model…")
def forecast(ticker: str, use_view: bool, horizon: int, drift: float, target_vol: float | None,
             through: date) -> PriceForecast:
    model = monitor_model(ticker, implied_vol(ticker, through) if use_view else None)
    return forecast_prices(closes(ticker, through), horizon, forecaster=model, n_samples=20_000,
                           annual_drift=drift, target_vol=target_vol, random_state=0)


@st.cache_data(ttl="12h", show_spinner="Scoring every past forecast (about 15-30 s the first time)…")
def history(ticker: str, use_view: bool, through: date) -> pd.DataFrame:
    iv = implied_vol(ticker, through) if use_view else None
    return track_record(closes(ticker, through), track_models(ticker, iv), start=TRACK_START)


def pct(x: float, signed: bool = True) -> str:
    return (f"{x:+.1%}" if signed else f"{x:.1%}").replace("-", "−")


# ----------------------------------------------------------------- controls
st.title("Risk monitor")
st.caption("Forecast distribution of returns from the last close: GJR-GARCH with skewed-t shocks, plus an "
           "implied-vol view where one has been validated. Every number is checked against what happened in "
           "the track record below.")

with st.container(horizontal=True, vertical_alignment="bottom"):
    ticker = st.text_input("Ticker", value="SPY", max_chars=12, key="ticker").strip().upper()
    horizon = st.segmented_control("Horizon (trading days)", [5, 10, 21, 63], default=21, key="horizon") or 21
    has_view = ticker in IV_VIEWS
    use_view = st.toggle(f"Implied-vol view ({IV_VIEWS[ticker][0].lstrip('^')})" if has_view else "Implied-vol view",
                         value=has_view, disabled=not has_view, key="view",
                         help="Blend the 21-day variance 50/50 with the asset's implied-vol index. Only offered "
                              f"where it was validated: {', '.join(IV_VIEWS)}.")
    drift = st.number_input("Drift view, % a year", value=0.0, step=1.0, min_value=-50.0, max_value=50.0,
                            key="drift", help="Default 0: no directional call. Trailing means were tested and "
                                              "helped only assets that happened to trend.")
    tvol = st.number_input("Volatility view, % a year", value=0.0, step=1.0, min_value=0.0, max_value=300.0,
                           key="tvol", help="0 = the model's own forecast. Otherwise the whole distribution is "
                                            "scaled so its volatility over the horizon matches your view.")

if not ticker:
    st.stop()
through = last_final_session()
try:
    fc = forecast(ticker, use_view, int(horizon), drift / 100, tvol / 100 or None, through)
except Exception as exc:  # noqa: BLE001 - bad ticker, too little history, network
    st.error(f"Couldn't forecast {ticker}: {exc}", title="No forecast")
    st.stop()

close = closes(ticker, through)
h = fc.horizon
risk = risk_table(fc, sorted({x for x in (1, 5, 10, 21, h) if x <= h})).set_index("horizon")
realised = float(np.log(close).diff().iloc[-21:].std() * np.sqrt(252))

# ----------------------------------------------------------------- headline
st.caption(f"{ticker} closed at {fc.last_price:,.2f} on {fc.as_of:%a %b %d, %Y} · model {fc.model}")
with st.container(horizontal=True):
    st.metric("1-day VaR 99%", pct(-risk.loc[1, "var_99"], False), f"ES {pct(-risk.loc[1, 'es_99'], False)}",
              help="Loss the next session should exceed only 1% of the time; ES is the average loss beyond it",
              **INFO)
    st.metric("5-day VaR 95%", pct(-risk.loc[5, "var_95"], False), f"ES {pct(-risk.loc[5, 'es_95'], False)}",
              **INFO)
    st.metric(f"{h}-day volatility", pct(risk.loc[h, "vol_ann"], False), f"last 21 days {pct(realised, False)}",
              help="Forecast volatility of the return over the horizon, annualised", **INFO)
    st.metric(f"P(up) in {h} days", pct(risk.loc[h, "p_up"], False),
              f"median {pct(risk.loc[h, 'median'] / fc.last_price - 1)}", **INFO)

left, right = st.columns([1.6, 1])
with left, st.container(border=True):
    st.markdown(f"**Last 6 months and the next {h} sessions**")
    st.plotly_chart(fan_chart(close.iloc[-126:], fc.bands()), width="stretch",
                    config={"displayModeBar": False})
with right, st.container(border=True):
    st.markdown(f"**{h}-day return distribution**")
    rets = fc.prices[:, -1] / fc.last_price - 1
    st.plotly_chart(return_histogram(rets, risk.loc[h, "var_95"], risk.loc[h, "var_99"]), width="stretch",
                    config={"displayModeBar": False})

with st.container(border=True):
    table = pd.DataFrame({
        "Horizon": [f"{x} day{'s' * (x > 1)}" for x in risk.index],
        "Date": [fc.dates[x - 1].strftime("%b %d") for x in risk.index],
        "Low (5%)": risk["lo_90"].to_numpy(), "Median": risk["median"].to_numpy(),
        "High (95%)": risk["hi_90"].to_numpy(),
        "VaR 99%": -risk["var_99"].to_numpy() * 100, "VaR 95%": -risk["var_95"].to_numpy() * 100,
        "ES 99%": -risk["es_99"].to_numpy() * 100, "Volatility": risk["vol_ann"].to_numpy() * 100,
        "P(up)": risk["p_up"].to_numpy() * 100,
    })
    money, share = st.column_config.NumberColumn(format="%.2f"), st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(table, hide_index=True, column_config={
        "Low (5%)": money, "Median": money, "High (95%)": money, "VaR 99%": share, "VaR 95%": share,
        "ES 99%": st.column_config.NumberColumn(format="%.1f%%", help="Average loss on the worst 1% of outcomes"),
        "Volatility": share, "P(up)": share})
    notes = ["Prices: 90% of simulated outcomes fall between Low and High. Losses are simple returns from the "
             "last close. Dates skip weekends but not exchange holidays."]
    if use_view:
        model = monitor_model(ticker, implied_vol(ticker, through))
        if isinstance(model, ImpliedVolForecaster):
            iv = implied_vol(ticker, through)
            notes.append(f"{IV_VIEWS[ticker][0].lstrip('^')} last {iv.iloc[-1] * 100:.1f} on "
                         f"{iv.index[-1]:%b %d}: the view pulls GJR's 21-day variance halfway to it.")
    if drift or tvol:
        notes.append("Your drift / volatility view is applied, so the track record below (model only) does not "
                     "describe these exact numbers.")
    st.caption(" ".join(notes))

# ----------------------------------------------------------------- track record
st.subheader("Track record")
st.caption(f"Every trading day since {TRACK_START[:4]}, the model was fitted on the two years before it (refit "
           "every 21 days) and its forecast was scored against the next 1, 5 and 21 days.")
try:
    rec = history(ticker, use_view, through)
except Exception as exc:  # noqa: BLE001
    st.error(f"Couldn't score past forecasts: {exc}", title="No track record")
    st.stop()
base = "gjr" if (rec["model"] == "gjr").any() else None
sc = score_track(rec, "augury", baseline=base)
if not sc.get("n"):
    st.info(f"Not enough history to score {ticker} yet (needs two years before the first forecast).")
    st.stop()

v1 = sc["var"]["0.01"]
c90 = sc["coverage_90"]
with st.container(horizontal=True):
    st.metric("1% VaR breaches, last 250 days", f"{v1['recent_breaches']}",
              f"{v1['recent_breaches'] - v1['recent_expected']:+.1f} vs expected · {v1['recent_zone']} zone",
              delta_color="inverse", border=True)
    st.metric(f"1% VaR breach rate since {pd.Timestamp(sc['since']):%Y}", f"{v1['rate']:.2%}",
              f"Kupiec p {v1['kupiec_p']:.2f} · clustering p {v1['christoffersen_p']:.2f}", **INFO)
    st.metric("90% range hit rate, 1 day", pct(c90[1], False),
              f"5 days {pct(c90[5], False)} · 21 days {pct(c90[21], False)}", **INFO)
    if sc.get("skill"):
        st.metric("Skill vs plain GJR, 1 day", pct(sc["skill"][1]), f"21 days {pct(sc['skill'][21])}",
                  help="CRPS improvement from the implied-vol view (> 0 = better)", **INFO)
    else:
        st.metric("Forecasts scored", f"{sc['n']:,}", f"since {pd.Timestamp(sc['since']):%b %Y}", **INFO)

one = rec[(rec["model"] == "augury") & (rec["horizon"] == 1)].sort_values("origin")
outcome = pd.DatetimeIndex(one["origin"]).to_series().shift(-1)
outcome.iloc[-1] = one["origin"].iloc[-1] + pd.offsets.BDay(1)
one = one.assign(date=outcome.to_numpy(), ret=np.expm1(one["y"]), v1=np.expm1(one["var_1"]),
                 v5=np.expm1(one["var_5"]))

left, right = st.columns([1.6, 1])
with left, st.container(border=True):
    st.markdown("**Daily returns against the VaR forecast from the night before**")
    win = st.segmented_control("Window", list(WINDOWS), default="1y", key="window",
                               label_visibility="collapsed") or "1y"
    d = one if WINDOWS[win] is None else one.iloc[-WINDOWS[win]:]
    st.plotly_chart(track_chart(d), width="stretch", config={"displayModeBar": False})
    st.caption(f"Red dots broke the 99% VaR: {int(d['hit_1'].sum())} in this window, where a calibrated model "
               f"would average {len(d) * 0.01:.1f}.")
with right, st.container(border=True):
    st.markdown("**Calibration of the 1-day forecasts**")
    st.plotly_chart(plot_pit_histogram(one["pit"].to_numpy()), width="stretch",
                    config={"displayModeBar": False})
    st.caption("Where each outcome fell in its forecast distribution. Flat = calibrated; high at both ends = "
               "ranges too narrow; a hump in the middle = too wide.")
