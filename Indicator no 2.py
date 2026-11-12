import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize

# 1. Interactive Configuration
print("=== Institutional Multi-Factor Walk-Forward Statistical Arbitrage ===")
stock_input = input("Enter portfolio stocks (default BBCA.JK, ANTM.JK, ADRO.JK, TLKM.JK, BBRI.JK): ").strip()
stocks = [t.strip().upper() for t in stock_input.split(",")] if stock_input else ["BBCA.JK", "ANTM.JK", "ADRO.JK", "TLKM.JK", "BBRI.JK"]

factor_input = input("Enter macro/market factors (first = benchmark index, default ^JKSE, IDR=X, GC=F): ").strip()
factors = [t.strip().upper() for t in factor_input.split(",")] if factor_input else ["^JKSE", "IDR=X", "GC=F"]

ridge_lambda = float(input("Enter Ridge penalty lambda (default 0.01): ").strip() or 0.01)
tx_cost_bps = float(input("Enter round-trip transaction cost in bps (default 20 = 0.20%): ").strip() or 20) / 10000.0

# 2. Download 3 Years of Historical Data
all_tickers = list(set(stocks + factors))
print(f"\nFetching 3-year market data for {len(all_tickers)} instruments...")
raw_data = yf.download(all_tickers, period="3y", progress=False)["Close"]

# Calculate daily returns per ticker before dropping holiday gaps
returns = raw_data.pct_change(fill_method=None)

# 3. Eliminate Look-Ahead Bias: Lag non-Jakarta (.JK / ^JKSE) factors by 1 trading day
for f in factors:
    if not (f.endswith(".JK") or f == "^JKSE"):
        returns[f] = returns[f].shift(1)

returns = returns.dropna()

# 4. Analytical Optimizer (SLSQP Quadratic Programming)
def solve_neutral_weights(exp_ret, cov, m_betas):
    n = len(exp_ret)
    
    def neg_sharpe(w):
        ret = np.dot(w, exp_ret) * 252
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w))) * np.sqrt(252)
        return -(ret / vol) if vol > 1e-8 else 1e6

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w)},              # Dollar-Neutral: Sum(Longs) + Sum(Shorts) = 0
        {"type": "eq", "fun": lambda w: np.dot(w, m_betas)},     # Beta-Neutral: Portfolio Market Beta = 0
        {"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0} # 100% Gross Leverage: Sum(|w|) = 1
    ]
    bounds = [(-0.40, 0.40) for _ in range(n)]  # Max 40% position cap per stock
    
    # Initialize with small alternating long/short weights to aid convergence
    w0 = np.array([0.2 if i % 2 == 0 else -0.2 for i in range(n)])
    w0 /= np.sum(np.abs(w0))
    
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 300})
    if res.success and np.sum(np.abs(res.x)) > 0.1:
        return res.x / np.sum(np.abs(res.x))
    return np.zeros(n)

# 5. Rolling Walk-Forward Engine (Monthly Rebalancing with Transaction Costs)
lookback = 252          # 1-year rolling training window
rebalance_freq = 21     # Rebalance every 21 trading days (~1 month)

oos_returns = []
oos_dates = []
current_w = np.zeros(len(stocks))
latest_betas = None
latest_r2 = None
total_turnover = 0.0

for t in range(lookback, len(returns)):
    # Re-estimate Ridge Regression & Rebalance every 21 trading days using ONLY past data
    if (t - lookback) % rebalance_freq == 0:
        win = returns.iloc[t - lookback : t]
        Y_w = win[stocks].values
        X_raw = win[factors].values
        n_rows, n_factors = X_raw.shape
        X_w = np.hstack([np.ones((n_rows, 1)), X_raw])
        
        # Ridge Regression: Beta = (X^T X + lambda * I)^(-1) X^T Y (unpenalized intercept)
        pen = ridge_lambda * np.eye(n_factors + 1)
        pen[0, 0] = 0.0
        b_w = np.linalg.inv(X_w.T @ X_w + pen) @ X_w.T @ Y_w
        
        # Track R-squared of latest window
        Y_hat = X_w @ b_w
        ss_res = np.sum((Y_w - Y_hat) ** 2, axis=0)
        ss_tot = np.sum((Y_w - np.mean(Y_w, axis=0)) ** 2, axis=0)
        latest_r2 = 1 - (ss_res / ss_tot)
        latest_betas = b_w
        
        new_w = solve_neutral_weights(win[stocks].mean().values, win[stocks].cov().values, b_w[1, :])
        if np.sum(np.abs(new_w)) > 0:
            turnover = np.sum(np.abs(new_w - current_w))
            total_turnover += turnover
            cost_drag = turnover * tx_cost_bps
            current_w = new_w
        else:
            cost_drag = 0.0
    else:
        cost_drag = 0.0

    # Realized out-of-sample return on day t net of transaction friction
    day_ret = np.dot(returns[stocks].iloc[t].values, current_w) - cost_drag
    oos_returns.append(day_ret)
    oos_dates.append(returns.index[t])

oos_series = pd.Series(oos_returns, index=oos_dates)
primary_market = factors[0]
bench_series = returns[primary_market].loc[oos_dates]

# 6. Institutional Performance & Risk Analytics
ann_ret = oos_series.mean() * 252
ann_vol = oos_series.std() * np.sqrt(252)
sharpe_ratio = ann_ret / ann_vol if ann_vol > 0 else 0.0

downside_std = oos_series[oos_series < 0].std() * np.sqrt(252)
sortino_ratio = ann_ret / downside_std if downside_std > 0 else 0.0

cum_strat = (1 + oos_series).cumprod()
cum_bench = (1 + bench_series).cumprod()
drawdown = (cum_strat / cum_strat.cummax()) - 1.0
max_dd = drawdown.min()

realized_corr = np.corrcoef(oos_series.values, bench_series.values)[0, 1]
net_factor_exposures = latest_betas[1:, :] @ current_w

# 7. Terminal Quantitative Tear Sheet
print(f"\n=== LATEST RIDGE MULTI-FACTOR EXPOSURES (TRAILING {lookback} DAYS) ===")
for idx, stock in enumerate(stocks):
    alpha_ann = latest_betas[0, idx] * 252
    print(f"Stock: {stock:8s} | R²: {latest_r2[idx]:.2%} | Ann. Alpha: {alpha_ann:+.2%} | {primary_market} Beta: {latest_betas[1, idx]:.3f}")

print("\n=== CURRENT REBALANCED DOLLAR & BETA-NEUTRAL BOOK ===")
for stock, weight in zip(stocks, current_w):
    side = "LONG " if weight >= 0 else "SHORT"
    print(f"{side} {stock:8s}: {weight:+.2%}")
print(f"Net Dollar Exposure: {np.sum(current_w):+.4f} | Gross Leverage: {np.sum(np.abs(current_w)):.2%}")

print(f"\n=== WALK-FORWARD OUT-OF-SAMPLE TEAR SHEET ({len(oos_series)} TRADING DAYS) ===")
print(f"Annualized Return (Net of Fees): {ann_ret:+.2%}")
print(f"Annualized Volatility:           {ann_vol:.2%}")
print(f"Sharpe Ratio (Rf=0):             {sharpe_ratio:.2f}")
print(f"Sortino Ratio:                   {sortino_ratio:.2f}")
print(f"Maximum Drawdown:                {max_dd:.2%}")
print(f"Realized Corr to {primary_market:6s}:         {realized_corr:+.4f}")

# 8. 4-Panel Dark-Mode Quantitative Tear Sheet Visualization
plt.style.use("dark_background")
fig = plt.figure(figsize=(15, 9), facecolor="#0d1117")
gs = gridspec.GridSpec(3, 2, height_ratios=[1.8, 0.8, 1.1], hspace=0.38, wspace=0.2)

ax_eq = fig.add_subplot(gs[0, :])
ax_dd = fig.add_subplot(gs[1, :], sharex=ax_eq)
ax_weights = fig.add_subplot(gs[2, 0])
ax_betas = fig.add_subplot(gs[2, 1])

for ax in (ax_eq, ax_dd, ax_weights, ax_betas):
    ax.set_facecolor("#161b22")
    ax.grid(True, linestyle="--", alpha=0.2, color="#8b949e")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Panel 1: Cumulative Equity Curve
ax_eq.plot(cum_strat.index, cum_strat, color="#3fb950", linewidth=2.2, label=f"Dollar/Beta-Neutral L/S (Net Sharpe: {sharpe_ratio:.2f} | Sortino: {sortino_ratio:.2f})")
ax_eq.plot(cum_bench.index, cum_bench, color="#8b949e", linewidth=1.5, linestyle="--", label=f"Benchmark ({primary_market})")
ax_eq.axhline(1.0, color="#30363d", linewidth=1)
ax_eq.set_title(
    f"Walk-Forward Statistical Arbitrage Tear Sheet ({len(oos_series)} OOS Days)  |  Ann. Ret: {ann_ret:+.1%}  |  Max DD: {max_dd:.1%}  |  Corr: {realized_corr:+.2f}",
    fontsize=12.5, fontweight="bold", color="#f0f6fc", loc="left", pad=10
)
ax_eq.set_ylabel("Cumulative Growth", color="#c9d1d9", fontweight="bold")
ax_eq.legend(loc="upper left", frameon=True, facecolor="#0d1117", edgecolor="#30363d", fontsize=9.5)
ax_eq.tick_params(labelbottom=False)

# Panel 2: Underwater Drawdown Curve
ax_dd.fill_between(drawdown.index, drawdown * 100, 0, color="#da3633", alpha=0.35)
ax_dd.plot(drawdown.index, drawdown * 100, color="#f85149", linewidth=1.1, label=f"Strategy Drawdown (Max: {max_dd:.2%})")
ax_dd.set_ylabel("Drawdown (%)", color="#c9d1d9", fontweight="bold")
ax_dd.legend(loc="lower left", frameon=True, facecolor="#0d1117", edgecolor="#30363d", fontsize=8.5)

# Panel 3: Current Rebalanced Long/Short Weights
weight_colors = ["#2ea043" if w >= 0 else "#da3633" for w in current_w]
ax_weights.barh(stocks, current_w * 100, color=weight_colors, alpha=0.85, height=0.55)
ax_weights.axvline(0, color="#8b949e", linewidth=1)
ax_weights.set_title("Current Rebalanced Allocation (%)", fontsize=11, fontweight="bold", color="#f0f6fc")
ax_weights.set_xlabel("Portfolio Weight (%)  [Sum = 0% Dollar-Neutral]", color="#c9d1d9", fontsize=9)

# Panel 4: Net Factor Exposures
beta_colors = ["#58a6ff" if abs(b) <= 0.02 else "#f0883e" for b in net_factor_exposures]
ax_betas.bar(factors, net_factor_exposures, color=beta_colors, alpha=0.85, width=0.45)
ax_betas.axhline(0, color="#8b949e", linewidth=1)
ax_betas.set_title("Net Portfolio Factor Exposures (Beta)", fontsize=11, fontweight="bold", color="#f0f6fc")
ax_betas.set_ylabel("Net Beta", color="#c9d1d9", fontsize=9)

plt.tight_layout()
plt.show()
