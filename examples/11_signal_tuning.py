"""Demo: Signal weight tuning on real market data.

Walk-forward approach:
  1. Download SPY data (2020-2026) — covers COVID crash, recovery,
     2022 bear market, 2023-2024 bull, and recent downturn.
  2. Pre-compute walk-forward forecasts with EnsembleForecaster.
  3. Baseline: tune all 9 parameters (original run).
  4. Regularised: add L2 penalty to keep weights near defaults.
  5. Weights-only: fix threshold & momentum, tune only 6 weights.
  6. Cross-validation: temporal expanding-window CV across all snapshots.
  7. Multi-ticker: pool SPY + QQQ + IWM for more training data.

Phases 1-3 reproduce the original baseline.
Phases 4-6 reuse pre-computed snapshots (fast).
Phase 7 downloads extra tickers (moderate).
"""

import sys
import time

import numpy as np
import yfinance as yf

from synthquant.strategy.signal_tuner import SignalTuner

# ── Log to file and stdout simultaneously ─────────────────────────────────────
_LOG_PATH = "_tuning_output.txt"
_logfile = open(_LOG_PATH, "w", encoding="utf-8")
_orig_print = print


def print(*args, **kwargs):
    kwargs.pop("flush", None)
    _orig_print(*args, **kwargs, flush=True)
    _orig_print(*args, file=_logfile, flush=True)


# ── Parameters ────────────────────────────────────────────────────────────────

TICKER = "SPY"
EXTRA_TICKERS = ["QQQ", "IWM"]
START = "2020-01-01"
END = "2026-12-31"
LOOKBACK = 504        # 2 years of history per forecast
HORIZON = 21          # 21-day forecast
STEP = 21             # forecast every 21 trading days
N_PATHS = 5000
TRAIN_FRAC = 0.65
TUNE_METRIC = "sharpe"
MAX_ITER = 80
MAX_ITER_FAST = 40      # fewer iters for comparative phases

# ── Helpers ───────────────────────────────────────────────────────────────────


def _download(ticker: str) -> tuple[np.ndarray, np.ndarray]:
    hist = yf.Ticker(ticker).history(start=START, end=END)
    prices = hist["Close"].values
    returns = np.diff(np.log(prices))
    returns = returns[np.isfinite(returns)]
    return returns, prices


def _fmt_row(label, m, total):
    return (
        f"    {label:<22} {m['sharpe']:>7.2f} {m['hit_rate']:>5.1%}"
        f" {m['info_ratio']:>6.2f} {m['n_trades']:>4}/{total:<2}"
        f" {m['total_pnl']:>+8.2%} {m['avg_ret_long']:>+8.2%}"
    )


def _fmt_header():
    h = f"    {'':>22} {'Sharpe':>7} {'Hit':>6} {'IC':>6} {'Trades':>7} {'PnL':>8} {'AvgLong':>8}"
    s = f"    {'':>22} {'---':>7} {'---':>6} {'---':>6} {'---':>7} {'---':>8} {'---':>8}"
    return h + "\n" + s


# ── Download data ─────────────────────────────────────────────────────────────

print("=" * 78)
print(f"  SIGNAL WEIGHT TUNING - {TICKER}")
print(f"  Period: {START} to present")
print(f"  Lookback: {LOOKBACK}d | Horizon: {HORIZON}d | Step: {STEP}d")
print(f"  Paths: {N_PATHS} | Metric: {TUNE_METRIC}")
print("=" * 78)

returns, prices = _download(TICKER)
print(f"\n  Data: {len(returns)} trading days")
print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")

# ── Phase 1: Walk-forward forecast pre-computation ───────────────────────────

print(f"\n{'-' * 78}")
print("  Phase 1: Walk-forward forecast generation")
print(f"{'-' * 78}")

tuner = SignalTuner(
    n_paths=N_PATHS,
    horizon=HORIZON,
    lookback=LOOKBACK,
    step=STEP,
)

t0 = time.time()


def progress(current, total, info):
    elapsed = time.time() - t0
    rate = current / elapsed if elapsed > 0 else 0
    eta = (total - current) / rate if rate > 0 else 0
    print(f"    [{current:>3}/{total}] {info}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")


snapshots = tuner.prepare(returns, prices, progress_fn=progress)
elapsed = time.time() - t0

print(f"\n  Generated {len(snapshots)} snapshots in {elapsed:.0f}s")

fwd_rets = np.array([s.actual_forward_return for s in snapshots])
print(f"  Forward returns: mean={np.mean(fwd_rets):+.2%}, std={np.std(fwd_rets):.2%}")
print(f"    Positive periods: {(fwd_rets > 0).sum()}")
print(f"    Negative periods: {(fwd_rets < 0).sum()}")
print(f"    Max gain:  {fwd_rets.max():+.2%}")
print(f"    Max loss:  {fwd_rets.min():+.2%}")

# ── Train/test split ─────────────────────────────────────────────────────────

split_idx = int(len(snapshots) * TRAIN_FRAC)
train_snaps = snapshots[:split_idx]
test_snaps = snapshots[split_idx:]

print(f"\n  Train: {len(train_snaps)} snapshots (first {TRAIN_FRAC:.0%})")
print(f"  Test:  {len(test_snaps)} snapshots (last {1 - TRAIN_FRAC:.0%})")

train_fwd = np.array([s.actual_forward_return for s in train_snaps])
test_fwd = np.array([s.actual_forward_return for s in test_snaps])
print(f"  Train fwd returns: mean={np.mean(train_fwd):+.2%}, pos={int((train_fwd > 0).sum())}, neg={int((train_fwd < 0).sum())}")
print(f"  Test  fwd returns: mean={np.mean(test_fwd):+.2%}, pos={int((test_fwd > 0).sum())}, neg={int((test_fwd < 0).sum())}")

# ── Phase 2: Default weight evaluation ───────────────────────────────────────

print(f"\n{'-' * 78}")
print("  Phase 2: Default weight evaluation")
print(f"{'-' * 78}")

default_train = tuner.evaluate(train_snaps)
default_test = tuner.evaluate(test_snaps)

fmt = "    {:<15} Sharpe={:>6.2f}  Hit={:>5.1%}  IC={:>6.2f}  Trades={:>2}/{:<2}  PnL={:>+7.2%}"
print(fmt.format("Train (default)", default_train["sharpe"], default_train["hit_rate"],
                  default_train["info_ratio"], default_train["n_trades"], len(train_snaps),
                  default_train["total_pnl"]))
print(fmt.format("Test  (default)", default_test["sharpe"], default_test["hit_rate"],
                  default_test["info_ratio"], default_test["n_trades"], len(test_snaps),
                  default_test["total_pnl"]))

# ── Phase 3: Baseline tune (all 9 params, no regularisation) ─────────────────

print(f"\n{'-' * 78}")
print(f"  Phase 3: Baseline tuning (all 9 params, no regularisation)")
print(f"{'-' * 78}")

t1 = time.time()


def tune_progress(generation, best_value):
    elapsed = time.time() - t1
    print(f"    Gen {generation:>3}  best {TUNE_METRIC}={best_value:+.3f}  ({elapsed:.0f}s)", flush=True)


baseline_result = tuner.tune(
    train_snaps, metric=TUNE_METRIC, max_iter=MAX_ITER,
    progress_fn=tune_progress,
)

baseline_best = baseline_result["params"]
baseline_train = baseline_result["metrics"]
baseline_test = tuner.evaluate(test_snaps, baseline_best)

print(f"\n  Baseline tune completed in {time.time() - t1:.0f}s")

# ── Phase 4: Regularised tune (lambda=0.5) ───────────────────────────────────

print(f"\n{'-' * 78}")
print("  Phase 4: Regularised tuning (lambda=0.5)")
print(f"{'-' * 78}")

t2 = time.time()
REG_LAMBDA = 0.5


def reg_progress(generation, best_value):
    elapsed = time.time() - t2
    print(f"    Gen {generation:>3}  best {TUNE_METRIC}={best_value:+.3f}  ({elapsed:.0f}s)", flush=True)


reg_result = tuner.tune(
    train_snaps, metric=TUNE_METRIC, max_iter=MAX_ITER_FAST,
    regularization=REG_LAMBDA, progress_fn=reg_progress,
)

reg_best = reg_result["params"]
reg_train = reg_result["metrics"]
reg_test = tuner.evaluate(test_snaps, reg_best)

print(f"\n  Regularised tune completed in {time.time() - t2:.0f}s")

# ── Phase 5: Weights-only tune ───────────────────────────────────────────────

print(f"\n{'-' * 78}")
print("  Phase 5: Weights-only tuning (6 params, threshold+momentum fixed)")
print(f"{'-' * 78}")

t3 = time.time()


def wo_progress(generation, best_value):
    elapsed = time.time() - t3
    print(f"    Gen {generation:>3}  best {TUNE_METRIC}={best_value:+.3f}  ({elapsed:.0f}s)", flush=True)


wo_result = tuner.tune(
    train_snaps, metric=TUNE_METRIC, max_iter=MAX_ITER_FAST,
    tune_weights_only=True, progress_fn=wo_progress,
)

wo_best = wo_result["params"]
wo_train = wo_result["metrics"]
wo_test = tuner.evaluate(test_snaps, wo_best)

print(f"\n  Weights-only tune completed in {time.time() - t3:.0f}s")

# ── Phase 6: Temporal cross-validation (4-fold) ──────────────────────────────

print(f"\n{'-' * 78}")
print("  Phase 6: Temporal cross-validation (4-fold, weights-only + reg)")
print(f"{'-' * 78}")

t4 = time.time()


def cv_progress(fold, total, fold_result):
    elapsed = time.time() - t4
    oos = fold_result["oos_tuned"]
    print(
        f"    Fold {fold}/{total}  "
        f"train={fold_result['n_train']}  test={fold_result['n_test']}  "
        f"OOS sharpe={oos['sharpe']:+.2f}  hit={oos['hit_rate']:.1%}  "
        f"({elapsed:.0f}s)",
        flush=True,
    )


cv_result = tuner.cross_validate(
    snapshots,
    n_folds=4,
    metric=TUNE_METRIC,
    max_iter=MAX_ITER_FAST,
    regularization=REG_LAMBDA,
    tune_weights_only=True,
    progress_fn=cv_progress,
)

print(f"\n  Cross-validation completed in {time.time() - t4:.0f}s")
print(f"  Mean OOS (default):  Sharpe={cv_result['mean_oos_default']['sharpe']:+.2f}  Hit={cv_result['mean_oos_default']['hit_rate']:.1%}")
print(f"  Mean OOS (tuned):    Sharpe={cv_result['mean_oos_tuned']['sharpe']:+.2f}  Hit={cv_result['mean_oos_tuned']['hit_rate']:.1%}")

# ── Phase 7: Multi-ticker pooling ────────────────────────────────────────────

print(f"\n{'-' * 78}")
print(f"  Phase 7: Multi-ticker ({TICKER} + {', '.join(EXTRA_TICKERS)})")
print(f"{'-' * 78}")

t5 = time.time()

# Use fewer paths for multi-ticker walk-forward (speed over precision)
multi_tuner = SignalTuner(
    n_paths=2000,
    horizon=HORIZON,
    lookback=LOOKBACK,
    step=STEP,
)

datasets = {TICKER: (returns, prices)}
for t_name in EXTRA_TICKERS:
    print(f"    Downloading {t_name}...")
    r, p = _download(t_name)
    datasets[t_name] = (r, p)
    print(f"      {len(r)} trading days, ${p.min():.2f} - ${p.max():.2f}")


def multi_progress(ticker, current, total):
    elapsed = time.time() - t5
    print(f"    Preparing {ticker} ({current}/{total})  ({elapsed:.0f}s)", flush=True)


multi_snaps = multi_tuner.prepare_multi(datasets, progress_fn=multi_progress)
print(f"\n  Pooled {len(multi_snaps)} snapshots from {len(datasets)} tickers in {time.time() - t5:.0f}s")

# Tune on pooled data, test on SPY-only test set
t6 = time.time()


def multi_tune_progress(generation, best_value):
    elapsed = time.time() - t6
    print(f"    Gen {generation:>3}  best {TUNE_METRIC}={best_value:+.3f}  ({elapsed:.0f}s)", flush=True)


multi_result = multi_tuner.tune(
    multi_snaps, metric=TUNE_METRIC, max_iter=MAX_ITER_FAST,
    regularization=REG_LAMBDA, tune_weights_only=True,
    progress_fn=multi_tune_progress,
)

multi_best = multi_result["params"]
multi_train = multi_result["metrics"]
multi_test = tuner.evaluate(test_snaps, multi_best)

print(f"\n  Multi-ticker tune completed in {time.time() - t6:.0f}s")

# ── Comparison table ──────────────────────────────────────────────────────────

print(f"\n{'=' * 78}")
print("  COMPARISON: OOS PERFORMANCE ON SPY TEST SET")
print(f"{'=' * 78}")
print(_fmt_header())
print(_fmt_row("Default", default_test, len(test_snaps)))
print(_fmt_row("Baseline (9p)", baseline_test, len(test_snaps)))
print(_fmt_row("Regularised (9p)", reg_test, len(test_snaps)))
print(_fmt_row("Weights-only (6p)", wo_test, len(test_snaps)))
cv_oos = cv_result["mean_oos_tuned"]
print(
    f"    {'CV mean (4-fold)':<22} {cv_oos['sharpe']:>7.2f} {cv_oos['hit_rate']:>5.1%}"
    f" {cv_oos['info_ratio']:>6.2f}     -       -        -"
)
print(_fmt_row("Multi-ticker pool", multi_test, len(test_snaps)))

# ── Weight comparison ─────────────────────────────────────────────────────────

print(f"\n{'-' * 78}")
print("  WEIGHT COMPARISON")
print(f"{'-' * 78}")
default_params = SignalTuner._default_params()
header = f"    {'':>14} {'Default':>8} {'Base':>8} {'Reg':>8} {'W-only':>8} {'Multi':>8}"
print(header)
print("    " + "-" * 54)
for name in SignalTuner.WEIGHT_NAMES:
    k = f"w_{name}"
    print(
        f"    {name:<14} {default_params[k]:>8.3f} {baseline_best[k]:>8.3f}"
        f" {reg_best[k]:>8.3f} {wo_best[k]:>8.3f} {multi_best[k]:>8.3f}"
    )
print(
    f"    {'threshold':<14} {default_params['conviction_threshold']:>8.3f}"
    f" {baseline_best['conviction_threshold']:>8.3f}"
    f" {reg_best['conviction_threshold']:>8.3f}"
    f" {wo_best['conviction_threshold']:>8.3f}"
    f" {multi_best['conviction_threshold']:>8.3f}"
)
print(
    f"    {'mom_short':<14} {default_params['momentum_short']:>8.0f}"
    f" {baseline_best['momentum_short']:>8.0f}"
    f" {reg_best['momentum_short']:>8.0f}"
    f" {wo_best['momentum_short']:>8.0f}"
    f" {multi_best['momentum_short']:>8.0f}"
)
print(
    f"    {'mom_long':<14} {default_params['momentum_long']:>8.0f}"
    f" {baseline_best['momentum_long']:>8.0f}"
    f" {reg_best['momentum_long']:>8.0f}"
    f" {wo_best['momentum_long']:>8.0f}"
    f" {multi_best['momentum_long']:>8.0f}"
)

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'=' * 78}")
print("  SUMMARY")
print(f"{'=' * 78}")
results = {
    "Default": default_test,
    "Baseline (9p)": baseline_test,
    "Regularised (9p)": reg_test,
    "Weights-only (6p)": wo_test,
    "Multi-ticker pool": multi_test,
}
best_label = max(results, key=lambda k: results[k]["sharpe"])
print(f"  Best OOS Sharpe: {best_label} = {results[best_label]['sharpe']:.2f}")
print(f"  Best OOS Hit:    {max(results, key=lambda k: results[k]['hit_rate'])} = {max(r['hit_rate'] for r in results.values()):.1%}")

for label, m in results.items():
    delta = m["sharpe"] - default_test["sharpe"]
    tag = "+" if delta > 0 else ""
    print(f"    {label:<22} Sharpe={m['sharpe']:>+.2f}  delta={tag}{delta:.2f}")

print(f"\n  Total runtime: {time.time() - t0:.0f}s")
print(f"{'=' * 78}")
