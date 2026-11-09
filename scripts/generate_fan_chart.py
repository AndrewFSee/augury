"""Generate the README fan chart with the default forecaster.

Fits GJR-GARCH with skew-t innovations (``augury.default_forecaster``)
to the last two years of SPY daily returns and plots the forecast price
distribution over the next 21 trading days, after three months of history.
Saves a dark-themed PNG to docs/fan_chart.png.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yfinance as yf

from augury import forecast_prices

# ── Config ───────────────────────────────────────────────────────────
TICKER = "SPY"
HORIZON = 21  # trading days
HISTORY_DAYS = 63  # history shown left of the forecast
N_SAMPLES = 20_000
OUTPUT = "docs/fan_chart.png"

# ── Fetch data and forecast ──────────────────────────────────────────
print(f"Fetching {TICKER} data...")
df = yf.download(TICKER, period="5y", progress=False, auto_adjust=True)
close = df["Close"].squeeze().dropna()

print("Fitting GJR-GARCH skew-t and simulating...")
fc = forecast_prices(close, horizon=HORIZON, n_samples=N_SAMPLES, random_state=42)
assert fc.dates is not None and fc.as_of is not None
print(f"  {TICKER} last close ${fc.last_price:,.2f} ({fc.as_of:%Y-%m-%d})")
print(fc.summary().to_string(float_format=lambda x: f"{x:.4f}"))

# Prepend today's price so the fan starts at the last close
dates = pd.DatetimeIndex([fc.as_of]).append(fc.dates)
paths = np.column_stack([np.full(N_SAMPLES, fc.last_price), fc.prices])

# ── Percentile bands ─────────────────────────────────────────────────
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
levels = sorted({x for lo, hi, _, _ in percentile_bands for x in (lo, hi)})
pcts = dict(zip(levels, np.percentile(paths, levels, axis=0), strict=True))
median = np.percentile(paths, 50, axis=0)

# ── Plot ─────────────────────────────────────────────────────────────
print("Rendering fan chart...")
fig, ax = plt.subplots(figsize=(14, 6.5))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

history = close.iloc[-HISTORY_DAYS:]
ax.plot(history.index, history.values, color="#e0e0e0", linewidth=1.4, label="SPY close")

for lo, hi, color, label in percentile_bands:
    ax.fill_between(dates, pcts[lo], pcts[hi], color=color, alpha=0.85, label=label, linewidth=0)
ax.plot(dates, median, color="white", linewidth=1.2, alpha=0.7, linestyle="--")
ax.axvline(fc.as_of, color="#555555", linewidth=0.8, linestyle=":", alpha=0.8)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %#d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
ax.tick_params(colors="#cccccc", labelsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#444444")
ax.grid(axis="y", color="#333333", linewidth=0.5, alpha=0.5)

var95 = fc.value_at_risk(0.05)
ax.set_title(
    f"Forecasted {TICKER} Price Distribution — Next {HORIZON} Trading Days\n"
    f"GJR-GARCH skew-t  ·  {N_SAMPLES:,} paths  ·  from ${fc.last_price:,.2f} "
    f"({fc.as_of:%Y-%m-%d})  ·  21-day 95% VaR {var95:.1%}",
    color="#e0e0e0",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=5,
    fontsize=8.5,
    frameon=False,
    labelcolor="#cccccc",
)

plt.tight_layout()
fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
print(f"Saved to {OUTPUT}")
plt.close()
