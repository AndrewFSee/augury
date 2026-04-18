"""Generate a fan chart image for the README.

Fits the EnsembleForecaster on SPY daily returns and produces a
24-hour intraday probabilistic forecast at 5-minute intervals.
Saves a dark-themed fan chart PNG to docs/fan_chart.png.
"""

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from synthquant.ensemble_forecast import EnsembleForecaster

# ── Config ───────────────────────────────────────────────────────────
TICKER = "SPY"
LOOKBACK = 504
HORIZON_HOURS = 24.0
INTERVAL_SEC = 300  # 5 minutes
N_PATHS = 500
OUTPUT = "docs/fan_chart.png"

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
print("Generating 24h intraday forecast...")
result = forecaster.forecast_intraday(
    horizon_hours=HORIZON_HOURS,
    interval_seconds=INTERVAL_SEC,
    n_paths=N_PATHS,
    current_price=last_price,
    asset_type="equity",
)

paths = result.paths  # (N_PATHS, n_steps+1)
n_steps = paths.shape[1]

# ── Build time axis ──────────────────────────────────────────────────
# Start from next trading day 9:30 AM ET
start_time = datetime(last_date.year, last_date.month, last_date.day, 9, 30)
start_time += timedelta(days=1)
# Adjust for weekends
while start_time.weekday() >= 5:
    start_time += timedelta(days=1)

times = [start_time + timedelta(seconds=i * INTERVAL_SEC) for i in range(n_steps)]

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

# Formatting
ax.xaxis.set_major_formatter(mdates.DateFormatter("%#I %p"))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax.tick_params(colors="#cccccc", labelsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#444444")
ax.spines["bottom"].set_color("#444444")
ax.grid(axis="y", color="#333333", linewidth=0.5, alpha=0.5)

# Title
ax.set_title(
    f"Forecasted {TICKER} Price Distribution — Next 24 Hours\n"
    f"EnsembleForecaster  ·  {N_PATHS} paths  ·  5-min intervals  ·  "
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
