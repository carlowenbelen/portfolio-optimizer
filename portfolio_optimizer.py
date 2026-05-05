"""
🎯  Portfolio Optimizer — Markowitz Efficient Frontier
=====================================================
Computes the classic Markowitz efficient frontier for any basket of assets,
finds the maximum-Sharpe portfolio and the minimum-variance portfolio, and
produces a self-contained HTML report with the iconic frontier curve, the
capital market line, and allocation pie charts.

Why this exists
---------------
Modern Portfolio Theory (Markowitz, 1952) won a Nobel Prize. Its core idea:
for any expected return level, there's a portfolio that achieves it with the
lowest possible risk. The set of all those portfolios traces out the
"efficient frontier." Real money managers still use this as their starting
point — usually with extra constraints layered on top.

Quick start
-----------
    pip install -r requirements.txt
    python portfolio_optimizer.py --preset tech-giants
    python portfolio_optimizer.py --tickers AAPL MSFT GOOGL AMZN --fetch-live
    python portfolio_optimizer.py --csv my_returns.csv

Output
------
    HTML report (reports/<timestamp>.html) with:
      • Efficient frontier curve (risk × return)
      • Capital market line + risk-free rate
      • Allocation pie charts for max-Sharpe and min-variance portfolios
      • Stats table with annualized return, vol, Sharpe, weights

Author
------
Carl Owen E. Belen — https://github.com/YOUR-USERNAME
Companion to the Trading Strategy Monte Carlo simulator.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRADING_DAYS = 252
DEFAULT_RISK_FREE = 0.045  # ~4.5% annual, current US Treasury 1Y range


# ============================================================
# 📐  Inputs / outputs
# ============================================================
@dataclass
class OptimizerInput:
    tickers: list[str]
    returns: pd.DataFrame  # daily returns
    risk_free_rate: float = DEFAULT_RISK_FREE


@dataclass
class Portfolio:
    weights: np.ndarray
    expected_return: float       # annualized
    volatility: float            # annualized
    sharpe: float

    def as_dict(self, tickers: list[str]) -> dict:
        return {
            "weights": dict(zip(tickers, [float(w) for w in self.weights])),
            "expected_return": float(self.expected_return),
            "volatility": float(self.volatility),
            "sharpe": float(self.sharpe),
        }


# ============================================================
# 🔢  Math
# ============================================================
def annualize(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return annualized expected returns vector and covariance matrix."""
    mu = returns.mean().to_numpy() * TRADING_DAYS
    sigma = returns.cov().to_numpy() * TRADING_DAYS
    return mu, sigma


def portfolio_stats(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                    rf: float) -> tuple[float, float, float]:
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ sigma @ w))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def neg_sharpe(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, rf: float) -> float:
    ret, vol, _ = portfolio_stats(w, mu, sigma, rf)
    return -((ret - rf) / vol) if vol > 0 else 1e6


def variance(w: np.ndarray, sigma: np.ndarray) -> float:
    return float(w @ sigma @ w)


def find_max_sharpe(mu: np.ndarray, sigma: np.ndarray, rf: float) -> Portfolio:
    n = len(mu)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.full(n, 1.0 / n)
    res = minimize(neg_sharpe, x0, args=(mu, sigma, rf),
                   bounds=bounds, constraints=constraints, method="SLSQP")
    w = res.x
    ret, vol, sharpe = portfolio_stats(w, mu, sigma, rf)
    return Portfolio(weights=w, expected_return=ret, volatility=vol, sharpe=sharpe)


def find_min_variance(mu: np.ndarray, sigma: np.ndarray, rf: float) -> Portfolio:
    n = len(mu)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.full(n, 1.0 / n)
    res = minimize(variance, x0, args=(sigma,),
                   bounds=bounds, constraints=constraints, method="SLSQP")
    w = res.x
    ret, vol, sharpe = portfolio_stats(w, mu, sigma, rf)
    return Portfolio(weights=w, expected_return=ret, volatility=vol, sharpe=sharpe)


def efficient_frontier(mu: np.ndarray, sigma: np.ndarray, rf: float,
                       n_points: int = 50) -> list[Portfolio]:
    """For each target return level between min-var and max-mu, find the lowest-risk portfolio."""
    n = len(mu)
    bounds = [(0.0, 1.0)] * n
    min_var = find_min_variance(mu, sigma, rf)
    target_returns = np.linspace(min_var.expected_return, mu.max() * 0.999, n_points)

    frontier = []
    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]
        x0 = np.full(n, 1.0 / n)
        res = minimize(variance, x0, args=(sigma,),
                       bounds=bounds, constraints=constraints, method="SLSQP")
        if not res.success:
            continue
        w = res.x
        ret, vol, sharpe = portfolio_stats(w, mu, sigma, rf)
        frontier.append(Portfolio(weights=w, expected_return=ret,
                                  volatility=vol, sharpe=sharpe))
    return frontier


# ============================================================
# 📥  Data loading
# ============================================================
def load_returns_from_csv(path: str) -> pd.DataFrame:
    """Load a CSV of daily prices (date column + ticker columns) and return daily returns."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df.pct_change().dropna()


def fetch_live_returns(tickers: list[str], days: int = 750) -> pd.DataFrame:
    """Fetch real prices via yfinance. Optional dependency."""
    try:
        import yfinance as yf
    except ImportError:
        print("⚠️  yfinance not installed.  pip install yfinance  (or pass --csv).",
              file=sys.stderr)
        sys.exit(1)
    end = datetime.now()
    start = end - pd.Timedelta(days=days)
    print(f"  → fetching {len(tickers)} tickers from yfinance "
          f"({start:%Y-%m-%d} to {end:%Y-%m-%d}) …")
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if "Close" in df.columns.get_level_values(0):
        df = df["Close"]
    return df.pct_change().dropna()


def synthetic_returns(tickers: list[str], days: int = 750,
                      seed: int = 42) -> pd.DataFrame:
    """Generate plausible synthetic returns when no real data source is available.
    Uses geometric Brownian motion with realistic per-asset parameters."""
    rng = np.random.default_rng(seed)
    n = len(tickers)
    # Annualized expected return between 4–18%, vol between 12–45%
    annual_mu = rng.uniform(0.04, 0.18, n)
    annual_sigma = rng.uniform(0.12, 0.45, n)
    # Sample correlation matrix (low–medium correlation)
    rho = rng.uniform(0.10, 0.65, (n, n))
    rho = (rho + rho.T) / 2
    np.fill_diagonal(rho, 1.0)
    # Build covariance
    cov = np.outer(annual_sigma, annual_sigma) * rho / TRADING_DAYS
    daily_mu = annual_mu / TRADING_DAYS
    samples = rng.multivariate_normal(daily_mu, cov, size=days)
    dates = pd.bdate_range(end=datetime.now(), periods=days)
    return pd.DataFrame(samples, index=dates, columns=tickers)


# ============================================================
# 🎨  Charts
# ============================================================
def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def chart_frontier(frontier: list[Portfolio], max_s: Portfolio,
                   min_v: Portfolio, mu: np.ndarray, sigma: np.ndarray,
                   tickers: list[str], rf: float) -> str:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="#0f0f10")
    ax.set_facecolor("#0f0f10")

    # Plot the frontier
    fr_vols = [p.volatility for p in frontier]
    fr_rets = [p.expected_return for p in frontier]
    ax.plot(fr_vols, fr_rets, color="#a3e635", linewidth=2.2,
            label="Efficient frontier")

    # Plot individual assets
    asset_vols = np.sqrt(np.diag(sigma))
    ax.scatter(asset_vols, mu, c="#22d3ee", s=80, alpha=0.85,
               edgecolors="white", linewidth=1, zorder=3, label="Individual assets")
    for i, t in enumerate(tickers):
        ax.annotate(t, (asset_vols[i], mu[i]),
                    xytext=(8, 6), textcoords="offset points",
                    color="#ddd", fontsize=9)

    # Highlight max-Sharpe and min-variance
    ax.scatter(max_s.volatility, max_s.expected_return, c="#fbbf24",
               s=200, marker="*", edgecolors="white", linewidth=1.5,
               zorder=4, label=f"Max Sharpe ({max_s.sharpe:.2f})")
    ax.scatter(min_v.volatility, min_v.expected_return, c="#fb7185",
               s=180, marker="X", edgecolors="white", linewidth=1.5,
               zorder=4, label="Min variance")

    # Capital Market Line: y = rf + slope * x where slope = max_s.sharpe
    x_max = max(asset_vols.max(), max_s.volatility) * 1.15
    cml_x = np.linspace(0, x_max, 100)
    cml_y = rf + max_s.sharpe * cml_x
    ax.plot(cml_x, cml_y, color="#888", linestyle="--", linewidth=1,
            alpha=0.7, label="Capital market line")

    # Risk-free rate marker
    ax.scatter([0], [rf], c="white", s=60, marker="s", zorder=4,
               label=f"Risk-free ({rf:.2%})")

    ax.set_title("Efficient Frontier", color="white", fontsize=16, pad=12)
    ax.set_xlabel("Annualized volatility (risk)", color="#aaa")
    ax.set_ylabel("Annualized expected return", color="#aaa")
    ax.tick_params(colors="#aaa")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(True, alpha=0.1)
    ax.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="white",
              loc="lower right", fontsize=9)

    out = _fig_to_b64(fig)
    plt.close(fig)
    return out


def chart_pie(portfolio: Portfolio, tickers: list[str], title: str,
              color: str) -> str:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#0f0f10")
    ax.set_facecolor("#0f0f10")

    weights = portfolio.weights
    # Hide near-zero allocations from the pie
    mask = weights > 0.001
    show_weights = weights[mask]
    show_labels = [tickers[i] for i in range(len(tickers)) if mask[i]]

    palette = ["#22d3ee", "#a3e635", "#fbbf24", "#fb7185", "#c084fc",
               "#60a5fa", "#fb923c", "#34d399", "#f472b6", "#a78bfa"]
    colors = [palette[i % len(palette)] for i in range(len(show_labels))]

    wedges, texts, autotexts = ax.pie(
        show_weights, labels=show_labels, colors=colors,
        autopct="%1.0f%%", startangle=90,
        textprops={"color": "white", "fontsize": 10},
        wedgeprops={"edgecolor": "#0f0f10", "linewidth": 2},
    )
    ax.set_title(title, color=color, fontsize=14, pad=12)

    out = _fig_to_b64(fig)
    plt.close(fig)
    return out


# ============================================================
# 📄  HTML report
# ============================================================
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Portfolio Optimizer Report</title>
<style>
:root {{ --bg: #0f0f10; --card: #1a1a1a; --text: #f0f0f0; --muted: #888; --accent: #a3e635; }}
body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; margin: 0; line-height: 1.6; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 3rem 2rem 5rem; }}
h1 {{ font-size: 2.5rem; margin: 0 0 0.4rem; letter-spacing: -0.02em; }}
.subtitle {{ color: var(--muted); margin-bottom: 2.5rem; }}
.eyebrow {{ color: var(--accent); text-transform: uppercase; letter-spacing: 2px; font-size: 0.78rem; font-weight: 700; margin-bottom: 0.4rem; }}
.stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 2.5rem; }}
.stat {{ background: var(--card); border: 1px solid #222; border-radius: 12px; padding: 1.25rem 1.5rem; }}
.stat-label {{ color: var(--muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.4rem; }}
.stat-value {{ font-size: 2rem; font-weight: 700; line-height: 1; color: white; }}
.stat-value.green {{ color: var(--accent); }}
.stat-value.gold {{ color: #fbbf24; }}
.chart {{ background: var(--card); border: 1px solid #222; border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem; }}
.chart img {{ width: 100%; display: block; border-radius: 8px; }}
.chart-row {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1.5rem; }}
.chart-row .chart {{ margin: 0; }}
table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 12px; overflow: hidden; margin-bottom: 1.5rem; }}
th, td {{ padding: 0.85rem 1.25rem; text-align: left; border-bottom: 1px solid #222; }}
th {{ background: #161616; color: var(--accent); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.4px; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
footer {{ color: var(--muted); border-top: 1px solid #222; padding-top: 1.5rem; margin-top: 3rem; font-size: 0.85rem; }}
@media (max-width: 700px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="container">
  <div class="eyebrow">Portfolio Optimizer Report</div>
  <h1>Markowitz Efficient Frontier</h1>
  <div class="subtitle">{n_assets} assets · {n_obs:,} return observations · risk-free {rf:.2%} · generated {generated_at}</div>

  <div class="stat-grid">
    <div class="stat"><div class="stat-label">Max Sharpe</div><div class="stat-value gold">{max_sharpe:.2f}</div></div>
    <div class="stat"><div class="stat-label">Max Sharpe return</div><div class="stat-value green">{max_ret:+.1%}</div></div>
    <div class="stat"><div class="stat-label">Max Sharpe risk</div><div class="stat-value">{max_vol:.1%}</div></div>
    <div class="stat"><div class="stat-label">Min variance return</div><div class="stat-value">{min_ret:+.1%}</div></div>
    <div class="stat"><div class="stat-label">Min variance risk</div><div class="stat-value green">{min_vol:.1%}</div></div>
  </div>

  <div class="chart"><img src="data:image/png;base64,{chart_frontier}"></div>

  <div class="chart-row">
    <div class="chart"><img src="data:image/png;base64,{chart_pie_max}"></div>
    <div class="chart"><img src="data:image/png;base64,{chart_pie_min}"></div>
  </div>

  <h2 style="margin-top: 3rem;">Asset statistics (annualized)</h2>
  <table>
    <thead><tr><th>Ticker</th><th class="num">Return</th><th class="num">Volatility</th><th class="num">Max-Sharpe weight</th><th class="num">Min-var weight</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>

  <footer>
    Generated by <strong>Portfolio Optimizer</strong>. Long-only, fully invested. Past performance does not predict future results.
  </footer>
</div>
</body>
</html>
"""


def render_report(tickers, mu, sigma, frontier, max_s, min_v, rf, n_obs, output_path):
    print("  → rendering charts …")
    fr_chart = chart_frontier(frontier, max_s, min_v, mu, sigma, tickers, rf)
    pie_max = chart_pie(max_s, tickers, "Max-Sharpe allocation", "#fbbf24")
    pie_min = chart_pie(min_v, tickers, "Min-variance allocation", "#fb7185")

    asset_vols = np.sqrt(np.diag(sigma))
    rows = "".join(
        f"<tr><td>{t}</td>"
        f"<td class='num'>{mu[i]:+.1%}</td>"
        f"<td class='num'>{asset_vols[i]:.1%}</td>"
        f"<td class='num'>{max_s.weights[i]:.0%}</td>"
        f"<td class='num'>{min_v.weights[i]:.0%}</td></tr>"
        for i, t in enumerate(tickers)
    )

    html = HTML.format(
        n_assets=len(tickers),
        n_obs=n_obs,
        rf=rf,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        max_sharpe=max_s.sharpe,
        max_ret=max_s.expected_return,
        max_vol=max_s.volatility,
        min_ret=min_v.expected_return,
        min_vol=min_v.volatility,
        chart_frontier=fr_chart,
        chart_pie_max=pie_max,
        chart_pie_min=pie_min,
        rows=rows,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


# ============================================================
# 🎯  Presets
# ============================================================
PRESETS_FILE = Path(__file__).resolve().parent / "presets.json"


def load_presets() -> dict:
    if not PRESETS_FILE.exists():
        return {}
    return json.loads(PRESETS_FILE.read_text(encoding="utf-8"))


# ============================================================
# 🖨️  Terminal output
# ============================================================
class K:
    CY = "\033[96m"; G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
    M = "\033[35m"; B = "\033[94m"; W = "\033[97m"; DIM = "\033[2m"
    BOLD = "\033[1m"; END = "\033[0m"


def c(t, color):
    return f"{color}{t}{K.END}"


def print_summary(tickers, max_s, min_v, rf):
    print()
    print(c("═" * 68, K.DIM))
    print(c("  🎯  Portfolio Optimizer — Markowitz Efficient Frontier", K.BOLD + K.CY))
    print(c("═" * 68, K.DIM))
    print()
    print(c("  ★ Max-Sharpe portfolio", K.Y))
    print(f"    Expected return:    {max_s.expected_return:>+8.2%}")
    print(f"    Volatility:         {max_s.volatility:>8.2%}")
    print(c(f"    Sharpe ratio:       {max_s.sharpe:>8.3f}", K.G))
    print(f"    Allocation:")
    for t, w in sorted(zip(tickers, max_s.weights), key=lambda x: -x[1]):
        if w > 0.001:
            print(f"      {t:<10} {w:>6.1%}")
    print()
    print(c("  ✕ Min-variance portfolio", K.M))
    print(f"    Expected return:    {min_v.expected_return:>+8.2%}")
    print(c(f"    Volatility:         {min_v.volatility:>8.2%}", K.G))
    print(f"    Sharpe ratio:       {min_v.sharpe:>8.3f}")
    print(f"    Allocation:")
    for t, w in sorted(zip(tickers, min_v.weights), key=lambda x: -x[1]):
        if w > 0.001:
            print(f"      {t:<10} {w:>6.1%}")
    print()
    print(c("═" * 68, K.DIM))


# ============================================================
# 🚀  CLI
# ============================================================
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="Markowitz portfolio optimizer.")
    parser.add_argument("--preset", help="run a preset basket")
    parser.add_argument("--list-presets", action="store_true",
                        help="list built-in presets and exit")
    parser.add_argument("--tickers", nargs="+", help="ticker symbols to optimize")
    parser.add_argument("--csv", help="path to a CSV of daily prices "
                                       "(date column + ticker columns)")
    parser.add_argument("--fetch-live", action="store_true",
                        help="use yfinance to fetch real recent prices")
    parser.add_argument("--days", type=int, default=750,
                        help="how many trading days of data to use (default: 750)")
    parser.add_argument("--rf", type=float, default=DEFAULT_RISK_FREE,
                        help=f"annual risk-free rate (default: {DEFAULT_RISK_FREE:.1%})")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for synthetic data fallback")
    parser.add_argument("--no-report", action="store_true",
                        help="terminal output only, skip the HTML report")
    args = parser.parse_args()

    presets = load_presets()

    if args.list_presets:
        if not presets:
            print(c("No presets found.", K.R))
            sys.exit(0)
        print(c("📋  Available presets:", K.BOLD + K.CY))
        for key, val in presets.items():
            print(f"  {c(key, K.B):<26}  {val.get('description', '')}")
            print(f"    {c(', '.join(val['tickers']), K.DIM)}")
            print()
        sys.exit(0)

    # Resolve tickers
    if args.preset:
        if args.preset not in presets:
            print(c(f"⚠️  Unknown preset '{args.preset}'. Use --list-presets.", K.R))
            sys.exit(1)
        tickers = presets[args.preset]["tickers"]
        preset_name = presets[args.preset].get("description", args.preset)
    elif args.tickers:
        tickers = args.tickers
        preset_name = "Custom basket"
    elif args.csv:
        df = load_returns_from_csv(args.csv)
        tickers = list(df.columns)
        preset_name = f"From {args.csv}"
    else:
        print(c("⚠️  Provide --preset, --tickers, or --csv.", K.R))
        print(c("    Run --list-presets to see options.", K.DIM))
        sys.exit(1)

    # Resolve returns data
    if args.csv:
        returns = df  # already loaded above
    elif args.fetch_live:
        returns = fetch_live_returns(tickers, days=args.days)
    else:
        print(c("  → using synthetic data (pass --fetch-live for real prices)", K.DIM))
        returns = synthetic_returns(tickers, days=args.days, seed=args.seed)

    print(c(f"  → {len(tickers)} assets, {len(returns):,} return observations", K.DIM))

    # Compute
    mu, sigma = annualize(returns)
    print(c("  → solving for max-Sharpe …", K.DIM))
    max_s = find_max_sharpe(mu, sigma, args.rf)
    print(c("  → solving for min-variance …", K.DIM))
    min_v = find_min_variance(mu, sigma, args.rf)
    print(c("  → tracing efficient frontier …", K.DIM))
    frontier = efficient_frontier(mu, sigma, args.rf)

    print_summary(tickers, max_s, min_v, args.rf)

    if not args.no_report:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe = preset_name.lower().replace(" ", "-").replace("/", "-")[:50]
        out = Path("reports") / f"{timestamp}-{safe}.html"
        path = render_report(tickers, mu, sigma, frontier, max_s, min_v,
                             args.rf, len(returns), out)
        print(c(f"  📄 Report saved: {path.absolute()}", K.G))
        print()


if __name__ == "__main__":
    main()
