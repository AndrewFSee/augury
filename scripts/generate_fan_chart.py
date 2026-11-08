"""Generate a fan chart image for the README.

Fits the EnsembleForecaster on SPY daily returns and produces a
5 trading-day probabilistic forecast at 30-minute intervals
(market hours only: 9:30 AM – 4:00 PM ET).
Saves a dark-themed fan chart PNG to docs/fan_chart.png.
"""

from datetime import timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import yfinance as yf

from synthquant.ensemble_forecast import EnsembleForecaster

# ── Config ───────────────────────────────────────────────────────────
TICKER = "SPY"
LOOKBACK = 504
N_TRADING_DAYS = 5
INTERVAL_MIN = 30  # 30-minute intervals
N_PATHS = 500
OUTPUT = "docs/fan_chart.png"

# Market hours: 9:30 AM – 4:00 PM ET = 6.5 hours = 390 minutes
MARKET_OPEN_H, MARKET_OPEN_M = 9, 30
MARKET_CLOSE_H, MARKET_CLOSE_M = 16, 0
INTERVALS_PER_DAY = 390 // INTERVAL_MIN  # 13 intervals per day
TOTAL_INTERVALS = N_TRADING_DAYS * INTERVALS_PER_DAY

# ── Fetch data ───────────────────────────────────────────────────────
print(f"Fetching {TICKER} data...")
df = yf.download(TICKER, start="2023-01-01", progress=False)
close = df["Close"].squeeze().dropna()
returns = np.log(close / close.shift(1)).dropna().values
last_price = float(close.iloc[-1])
last_date = close.index[-1]

print(f"  {TICKER} last price: ${last_price:.2f}  ({last_date.strftime('%Y-%m-%d')})")
print(f"  Using last {LOOKBACK} returns for calibration")

# ── Fit ensemble forecaster ──────────────────────────────────────────
print("Fitting EnsembleForecaster...")
forecaster = EnsembleForecaster(
    detector="hmm",
    n_regimes=2,
    n_paths=N_PATHS,
    vol_model="GARCH",
    random_state=42,
    use_filtered=True,
)
forecaster.fit(returns[-LOOKBACK:])

# ── Generate intraday forecast ───────────────────────────────────────
# Use forecast_intraday for the full span, then filter to market hours
total_hours = N_TRADING_DAYS * 24.0
print(f"Generating {N_TRADING_DAYS}-day forecast ({TOTAL_INTERVALS} intervals)...")
result = forecaster.forecast_intraday(
    horizon_hours=total_hours,
    interval_seconds=INTERVAL_MIN * 60,
    n_paths=N_PATHS,
    current_price=last_price,
    asset_type="equity",
)

raw_paths = result.paths  # (N_PATHS, n_steps+1)

# ── Build time axis and filter to market hours ───────────────────────
# Start from next trading day 9:30 AM ET
start_date = last_date.to_pydatetime().replace(
    hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0, microsecond=0
)
start_date += timedelta(days=1)
# Skip weekends
while start_date.weekday() >= 5:
    start_date += timedelta(days=1)

# Build full time axis from raw paths
raw_times = [
    start_date + timedelta(minutes=i * INTERVAL_MIN)
    for i in range(raw_paths.shape[1])
]

# Filter to market hours only
market_indices = []
for i, t in enumerate(raw_times):
    market_open = t.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M)
    market_close = t.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M)
    if t.weekday() < 5 and market_open <= t <= market_close:
        market_indices.append(i)

# Limit to our target number of intervals
market_indices = market_indices[: TOTAL_INTERVALS + 1]
paths = raw_paths[:, market_indices]
times = [raw_times[i] for i in market_indices]

print(f"  {len(times)} data points across {N_TRADING_DAYS} trading days")

# ── Compute percentile bands ────────────────────────────────────────
percentile_bands = [
    (0.5, 5.0, "#e74c3c", "0.5–5 Percentile"),
    (5.0, 20.0, "#e67e22", "5–20 Percentile"),
    (20.0, 35.0, "#f39c12", "20–35 Percentile"),
    (35.0, 50.0, "#d4ac6e", "35–50 Percentile"),
    (50.0, 65.0, "#52b788", "50–65 Percentile"),
    (65.0, 80.0, "#2d9f6f", "65–80 Percentile"),
    (80.0, 95.0, "#1b7f5a", "80–95 Percentile"),
    (95.0, 99.5, "#0f5e3f", "95–99.5 Percentile"),
]

pcts = {}
for lo, hi, _, _ in percentile_bands:
    if lo not in pcts:
        pcts[lo] = np.percentile(paths, lo, axis=0)
    if hi not in pcts:
        pcts[hi] = np.percentile(paths, hi, axis=0)
median = np.percentile(paths, 50, axis=0)

# ── Plot ─────────────────────────────────────────────────────────────
print("Rendering fan chart...")
fig, ax = plt.subplots(figsize=(14, 6.5))

# Dark theme
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

# Fill bands from outside in (tail first)
for lo, hi, color, label in percentile_bands:
    ax.fill_between(
        times,
        pcts[lo],
        pcts[hi],
        color=color,
        alpha=0.85,
        label=label,
        linewidth=0,
    )

# Median line
ax.plot(times, median, color="white", linewidth=1.2, alpha=0.7, linestyle="--")

# Day separators — vertical lines at market open
for d in range(1, N_TRADING_DAYS):
    day_start = start_date + timedelta(days=d)
    while day_start.weekday() >= 5:
        day_start += timedelta(days=1)
    ax.axvline(day_start, color="#555555", linewidth=0.8, linestyle=":", alpha=0.6)

# Formatting — show day names
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %#m/%#d"))
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.tick_params(colors="#cccccc", labelsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#444444")
ax.spines["bottom"].set_color("#444444")
ax.grid(axis="y", color="#333333", linewidth=0.5, alpha=0.5)

# Title
ax.set_title(
    f"Forecasted {TICKER} Price Distribution — Next {N_TRADING_DAYS} Trading Days\n"
    f"EnsembleForecaster  ·  {N_PATHS} paths  ·  {INTERVAL_MIN}-min intervals  ·  "
    f"from ${last_price:,.2f} ({last_date.strftime('%Y-%m-%d')})",
    color="#e0e0e0",
    fontsize=13,
    fontweight="bold",
    pad=15,
)

# Legend
legend = ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=4,
    fontsize=8.5,
    frameon=False,
    labelcolor="#cccccc",
)

plt.tight_layout()
fig.savefig(
    OUTPUT,
    dpi=180,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
    edgecolor="none",
)
print(f"Saved to {OUTPUT}")
plt.close()
