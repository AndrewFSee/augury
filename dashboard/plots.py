"""Plotly figures for the augury dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError as e:
    raise ImportError(
        "Plotly is required for dashboard plots. Install with: pip install augury[dashboard]"
    ) from e

__all__ = ["fan_chart", "return_histogram", "track_chart", "plot_pit_histogram"]

TEAL, GREY, RED, AMBER = "#2BB5A6", "#9AA5B1", "#E4572E", "#F2C14E"
_LAYOUT = {"margin": {"l": 0, "r": 0, "t": 10, "b": 0},
           "legend": {"orientation": "h", "y": 1.1, "x": 0},
           "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"}


def fan_chart(history: pd.Series, bands: pd.DataFrame, height: int = 340) -> go.Figure:
    """Recent closes followed by the forecast's 50% / 90% price bands and median.

    Args:
        history: Recent closes, dated.
        bands: Output of ``PriceForecast.bands()`` with columns q5, q25, q50, q75, q95.
    """
    last_date, last = history.index[-1], float(history.iloc[-1])
    x = [last_date, *bands.index]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history, mode="lines", name="close",
                             line={"width": 1.6, "color": GREY},
                             hovertemplate="%{x|%b %d}: %{y:,.2f}<extra></extra>"))
    for lo, hi, alpha, name in (("q5", "q95", 0.22, "90% range"), ("q25", "q75", 0.42, "50% range")):
        fig.add_trace(go.Scatter(x=x, y=[last, *bands[hi]], mode="lines", line={"width": 0},
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=[last, *bands[lo]], mode="lines", line={"width": 0},
                                 fill="tonexty", fillcolor=f"rgba(43,181,166,{alpha})", name=name,
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=[last, *bands["q50"]], mode="lines", name="median",
                             line={"width": 2, "color": TEAL},
                             hovertemplate="%{x|%b %d}: %{y:,.2f}<extra>median</extra>"))
    fig.update_layout(height=height, hovermode="x unified", **_LAYOUT)
    return fig


def return_histogram(returns: np.ndarray, var_95: float, var_99: float, height: int = 300) -> go.Figure:
    """Histogram of simulated simple returns with the 95% / 99% VaR marked."""
    lo, hi = np.quantile(returns, [0.002, 0.998])  # drop the extreme tails rather than pile them at the edges
    counts, edges = np.histogram(returns[(returns >= lo) & (returns <= hi)], bins=80)
    mid = 0.5 * (edges[:-1] + edges[1:])
    colors = [RED if m <= var_99 else AMBER if m <= var_95 else GREY for m in mid]
    fig = go.Figure(go.Bar(x=mid, y=counts / counts.sum(), marker_color=colors, marker_line_width=0,
                           hovertemplate="%{x:+.1%}: %{y:.1%} of outcomes<extra></extra>"))
    for v, label, side in ((var_95, "VaR 95%", "top right"), (var_99, "VaR 99%", "top left")):
        fig.add_vline(x=v, line_dash="dot", line_width=1, annotation_text=f"{label}<br>{v:+.1%}",
                      annotation_position=side)
    fig.update_layout(height=height, bargap=0.05, showlegend=False,
                      xaxis={"tickformat": ".0%"}, yaxis={"tickformat": ".0%"}, **_LAYOUT)
    return fig


def track_chart(track: pd.DataFrame, height: int = 320) -> go.Figure:
    """Daily returns against the VaR forecast made the night before.

    Args:
        track: Columns ``date`` (outcome date), ``ret``, ``v1`` / ``v5`` (1% / 5% VaR) and
            ``hit_1`` (breach of the 1% VaR), all as simple returns.
    """
    hit = track["hit_1"].astype(bool).to_numpy()
    fig = go.Figure([
        go.Scatter(x=track["date"], y=track["ret"], mode="markers", name="daily return",
                   marker={"size": np.where(hit, 8, 4), "color": np.where(hit, RED, GREY)},
                   hovertemplate="%{x|%b %d, %Y}: %{y:+.2%}<extra></extra>"),
        go.Scatter(x=track["date"], y=track["v5"], mode="lines", name="95% VaR",
                   line={"width": 1, "color": AMBER}, hovertemplate="%{y:+.2%}<extra>95% VaR</extra>"),
        go.Scatter(x=track["date"], y=track["v1"], mode="lines", name="99% VaR",
                   line={"width": 1.4, "color": RED}, hovertemplate="%{y:+.2%}<extra>99% VaR</extra>"),
    ])
    fig.update_layout(height=height, yaxis={"tickformat": ".0%"}, hovermode="closest", **_LAYOUT)
    return fig


def plot_pit_histogram(pit_values: np.ndarray, n_bins: int = 10, height: int = 260) -> go.Figure:
    """PIT histogram: flat means calibrated; U-shaped = too narrow; hump = too wide."""
    counts, edges = np.histogram(pit_values, bins=n_bins, range=(0, 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    share = counts / max(counts.sum(), 1)
    fig = go.Figure(go.Bar(x=centers, y=share, width=0.9 / n_bins, marker_color=TEAL,
                           hovertemplate="PIT %{x:.2f}: %{y:.1%}<extra></extra>"))
    fig.add_hline(y=1 / n_bins, line_dash="dash", line_color=RED, annotation_text="calibrated")
    fig.update_layout(height=height, showlegend=False, xaxis={"title": "PIT", "range": [0, 1]},
                      yaxis={"tickformat": ".0%"}, **_LAYOUT)
    return fig
