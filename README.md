# 🎯 Portfolio Optimizer

Computes the **Markowitz efficient frontier** for any basket of assets, finds the **maximum-Sharpe portfolio** and the **minimum-variance portfolio**, and produces a self-contained HTML report with the iconic frontier curve, the capital market line, and allocation pie charts.

## Why I built it

I run a Monte Carlo simulator for trading strategies (the [companion repo](https://github.com/YOUR-USERNAME/trading-strategy-monte-carlo)). That tool tells you what *one* strategy will do under randomness. This tool answers a different question: **given a basket of assets, what's the best mix?**

Modern Portfolio Theory (Markowitz 1952) won a Nobel Prize for the answer: for any expected return target, there's a portfolio that achieves it with the lowest possible risk. The set of all those portfolios traces out the **efficient frontier**. Real money managers still start here, then layer constraints on top.

## What it does

1. Take a list of tickers (or a CSV of returns)
2. Compute the annualized expected return vector and covariance matrix
3. Solve two optimization problems with SciPy:
   - **Min-variance** — portfolio with the lowest possible volatility
   - **Max-Sharpe** — portfolio with the best risk-adjusted return
4. Trace the **efficient frontier** by solving for min-variance at each target return
5. Draw the **capital market line** (the line tangent to the frontier through the risk-free rate)
6. Render a self-contained HTML report

## Quick start

```bash
git clone https://github.com/YOUR-USERNAME/portfolio-optimizer.git
cd portfolio-optimizer
pip install -r requirements.txt

# See built-in presets (works offline with synthetic data)
python portfolio_optimizer.py --list-presets
python portfolio_optimizer.py --preset tech-giants

# Use real prices from yfinance
python portfolio_optimizer.py --tickers AAPL MSFT GOOGL AMZN --fetch-live

# Use your own CSV (date column + ticker columns of daily prices)
python portfolio_optimizer.py --csv my_prices.csv
```

The terminal prints a summary; an HTML report is saved to `reports/<timestamp>.html`:

```
══════════════════════════════════════════════════════════════════
  🎯  Portfolio Optimizer — Markowitz Efficient Frontier
══════════════════════════════════════════════════════════════════

  ★ Max-Sharpe portfolio
    Expected return:     +14.32%
    Volatility:           18.45%
    Sharpe ratio:          0.534
    Allocation:
      MSFT       38.2%
      NVDA       27.1%
      GOOGL      18.6%
      AAPL       11.4%
      AMZN        4.7%

  ✕ Min-variance portfolio
    Expected return:      +9.18%
    Volatility:           14.22%
    Sharpe ratio:          0.330
    Allocation:
      MSFT       42.1%
      AAPL       28.4%
      GOOGL      18.5%
      AMZN        7.2%
      NVDA        3.8%
```

## Built-in presets

| Preset | Description |
|---|---|
| `tech-giants` | Five US mega-cap tech names (AAPL, MSFT, GOOGL, AMZN, NVDA) |
| `magnificent-7` | The "Magnificent 7" that drove most of 2023–24 S&P 500 returns |
| `defensive` | Low-volatility consumer staples + healthcare (JNJ, KO, PG, WMT, PEP) |
| `all-weather` | Inspired by Ray Dalio's All Weather portfolio (VTI, TLT, IEF, GLD, DBC) |
| `crypto-mix` | Major crypto via spot ETFs and proxies (IBIT, FBTC, ETHE, COIN, MSTR) |

## CLI flags

| Flag | What it does |
|---|---|
| `--preset <name>` | Use a built-in basket |
| `--list-presets` | Show all built-in presets and exit |
| `--tickers AAPL MSFT …` | Custom basket |
| `--csv <path>` | Load daily prices from a CSV (date column + ticker columns) |
| `--fetch-live` | Use yfinance to fetch real recent prices (~3 years) |
| `--days <N>` | How many trading days of data to use (default: 750) |
| `--rf <rate>` | Annual risk-free rate (default: 4.5%) |
| `--seed <int>` | Random seed for synthetic-data fallback |
| `--no-report` | Terminal output only, skip HTML |

## How it works (the math)

For a portfolio with weights `w` over assets with mean returns vector `μ` and covariance matrix `Σ`:

| Quantity | Formula |
|---|---|
| Expected return | `μₚ = w'μ` |
| Variance | `σ²ₚ = w'Σw` |
| Sharpe ratio | `(μₚ − rf) / √σ²ₚ` |

**Min-variance optimization:**
```
minimize    w'Σw
subject to  Σwᵢ = 1
            wᵢ ≥ 0          (long-only)
```

**Max-Sharpe optimization:**
```
maximize   (w'μ − rf) / √(w'Σw)
subject to  Σwᵢ = 1
            wᵢ ≥ 0          (long-only)
```

**Efficient frontier:** for each target return `r*`, find the lowest-variance portfolio that achieves `r*`:
```
minimize    w'Σw
subject to  Σwᵢ = 1
            w'μ = r*
            wᵢ ≥ 0
```

The whole problem is a quadratic program with linear constraints — SciPy's `SLSQP` handles it cleanly.

## Honest caveats

- **Long-only, fully invested.** No shorts, no cash, no leverage. Easy to extend by relaxing the bounds and the sum-to-1 constraint.
- **Past performance ≠ future results.** Mean returns and covariances estimated from historical data are noisy and unstable. Real money managers use shrinkage estimators (Ledoit-Wolf), Black-Litterman views, and regime models to reduce this.
- **No transaction costs / slippage / taxes** modeled. Those matter a lot in real life.
- **Stationarity assumption.** The optimizer treats μ and Σ as stable. They aren't — correlations break down exactly when you most need them (during market stress).
- **Vanilla mean-variance is sensitive to inputs.** Tiny changes in μ can flip allocations dramatically. This is *the* known weakness of MPT.

For real money, treat this tool as the **starting point** for a conversation, not the answer.

## Tech

- **Python 3.10+**
- **NumPy + pandas** for vectorized math
- **SciPy** for the constrained quadratic optimization (`SLSQP`)
- **matplotlib** for charts
- **yfinance** (optional) for live data
- ~480 lines of well-organized code

## Possible extensions

- Add Black-Litterman with subjective views
- Implement Ledoit-Wolf covariance shrinkage
- Add CVaR (conditional value-at-risk) optimization
- Bootstrap the frontier — resample returns and show frontier uncertainty bands
- Allow short-selling (relax `wᵢ ≥ 0`)
- Add cardinality constraints (max N holdings)

## License

MIT — use it, fork it, ship something.

---

Built by **Carl Owen E. Belen** &middot; companion to my [Trading Strategy Monte Carlo simulator](https://github.com/YOUR-USERNAME/trading-strategy-monte-carlo) &middot; [Portfolio](https://github.com/carlowenbelen)
