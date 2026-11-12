import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 1. User Configuration
print("=== Quantitative Market Microstructure & Flow Dashboard ===")
ticker = input("Enter Ticker (e.g., NVDA, AAPL, BBCA.JK): ").upper().strip() or "NVDA"
period = input("Enter Period (1mo, 3mo, 6mo, 1y) [default: 6mo]: ").lower().strip() or "6mo"

# 2. Fetch & Clean Data
print(f"\nFetching market data & ownership structure for {ticker}...")
data = yf.download(ticker, period=period, progress=False)

if data.empty:
    print(f"Error: No data found for {ticker}. Check the ticker symbol and try again.")
    exit()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Fetch Institutional / Insider Ownership Breakdown (Available primarily for US/Global stocks)
stock_obj = yf.Ticker(ticker)
try:
    holders = stock_obj.major_holders
    if holders is not None and not holders.empty:
        print("\n--- Major Ownership Breakdown ---")
        print(holders)
except Exception:
    pass

# 3. Feature Engineering & Microstructure Signals
# Trend: 20 & 50 Simple Moving Averages
data["SMA_20"] = data["Close"].rolling(20).mean()
data["SMA_50"] = data["Close"].rolling(50).mean()

# Institutional Flow: Accumulation/Distribution (A/D) Line & Close Location Value (CLV)
hl_diff = (data["High"] - data["Low"]).replace(0, np.nan)
mf_mult = (((data["Close"] - data["Low"]) - (data["High"] - data["Close"])) / hl_diff).fillna(0)
data["CLV"] = mf_mult
data["AD_Line"] = (mf_mult * data["Volume"]).cumsum()

# Statistical Volume Anomaly (20-day Rolling Z-Score)
vol_mean = data["Volume"].rolling(20).mean()
vol_std = data["Volume"].rolling(20).std().replace(0, np.nan)
data["Vol_Z"] = ((data["Volume"] - vol_mean) / vol_std).fillna(0)

# Flow Classification (Wyckoff Volume Spread Analysis Proxy)
# Retail Panic Selling: Extreme volume spike + down day + price absorbed off the lows (CLV > -0.3)
data["Panic_Sell"] = (data["Vol_Z"] > 2.0) & (data["Close"] < data["Open"]) & (data["CLV"] > -0.3)

# Institutional Outflow (Distribution): High volume + down day + closes at the bottom (CLV <= -0.6)
data["Inst_Outflow"] = (data["Vol_Z"] > 1.2) & (data["Close"] < data["Open"]) & (data["CLV"] <= -0.6)

# Institutional Inflow (Accumulation): High volume + up day + closes near the top (CLV >= 0.6)
data["Inst_Inflow"] = (data["Vol_Z"] > 1.2) & (data["Close"] > data["Open"]) & (data["CLV"] >= 0.6)

# Order Flow Proxy: Buy vs. Sell Volume by Price Level (Volume Profile)
data["Buy_Vol"] = np.where(data["Close"] >= data["Open"], data["Volume"], 0)
data["Sell_Vol"] = np.where(data["Close"] < data["Open"], data["Volume"], 0)

bins = np.linspace(data["Low"].min(), data["High"].max(), 25)
data["Price_Bin"] = pd.cut(data["Close"], bins=bins)
profile = data.groupby("Price_Bin", observed=False)[["Buy_Vol", "Sell_Vol"]].sum()
bin_centers = [interval.mid for interval in profile.index]
poc_price = bin_centers[np.argmax(profile["Buy_Vol"] + profile["Sell_Vol"])]  # Point of Control

# Determine Recent Market Regime (Last 10 trading days)
recent = data.tail(10)
if recent["Inst_Outflow"].sum() > recent["Inst_Inflow"].sum():
    regime_text = "Regime (10D): Institutional Distribution"
    regime_color = "#f85149"
elif recent["Inst_Inflow"].sum() > recent["Inst_Outflow"].sum():
    regime_text = "Regime (10D): Institutional Accumulation"
    regime_color = "#3fb950"
elif recent["Panic_Sell"].sum() > 0:
    regime_text = "Regime (10D): Retail Capitulation / Absorption"
    regime_color = "#d29922"
else:
    regime_text = "Regime (10D): Neutral / Balanced Flow"
    regime_color = "#8b949e"

# 4. Professional Dark-Theme Dashboard Layout
plt.style.use("dark_background")
fig = plt.figure(figsize=(15, 8.5), facecolor="#0d1117")
gs = gridspec.GridSpec(2, 2, width_ratios=[3.6, 1], height_ratios=[2.3, 1], hspace=0.12, wspace=0.05)

ax_price = fig.add_subplot(gs[0, 0])
ax_vp = fig.add_subplot(gs[0, 1], sharey=ax_price)
ax_ad = fig.add_subplot(gs[1, 0], sharex=ax_price)

panel_bg = "#161b22"
for ax in (ax_price, ax_vp, ax_ad):
    ax.set_facecolor(panel_bg)
    ax.grid(True, linestyle="--", alpha=0.2, color="#8b949e")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# --- Panel 1: Price Action, Moving Averages & Flow Signals ---
ax_price.plot(data.index, data["Close"], color="#58a6ff", linewidth=1.8, label="Close Price", zorder=2)
ax_price.plot(data.index, data["SMA_20"], color="#f0883e", linewidth=1.2, linestyle="--", label="20 SMA", zorder=2)
ax_price.plot(data.index, data["SMA_50"], color="#bc8cff", linewidth=1.2, linestyle="--", label="50 SMA", zorder=2)
ax_price.axhline(poc_price, color="#f85149", linestyle=":", linewidth=1.3, label=f"POC ({poc_price:,.2f})", zorder=1)

# Plot Institutional & Retail Markers
inflow_days = data[data["Inst_Inflow"]]
outflow_days = data[data["Inst_Outflow"]]
panic_days = data[data["Panic_Sell"]]

ax_price.scatter(inflow_days.index, inflow_days["Low"] * 0.99, marker="^", color="#3fb950", s=75, zorder=5, label="Inst. Accumulation")
ax_price.scatter(outflow_days.index, outflow_days["High"] * 1.01, marker="v", color="#f85149", s=75, zorder=5, label="Inst. Outflow / Dump")
ax_price.scatter(panic_days.index, panic_days["Low"] * 0.985, marker="D", color="#d29922", s=60, zorder=5, label="Panic Sell / Absorption")

# Header & Automated Regime Badge
last_close = data["Close"].iloc[-1]
pct_change = ((last_close / data["Close"].iloc[0]) - 1) * 100
ax_price.set_title(
    f"{ticker}  |  Last: {last_close:,.2f} ({pct_change:+.2f}%)  |  Period: {period.upper()}",
    fontsize=13, fontweight="bold", color="#f0f6fc", loc="left", pad=12
)
ax_price.text(
    0.99, 0.95, regime_text, transform=ax_price.transAxes,
    fontsize=9.5, fontweight="bold", color=regime_color,
    ha="right", va="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d1117", edgecolor=regime_color, alpha=0.9)
)

ax_price.set_ylabel("Price", color="#c9d1d9", fontweight="bold")
ax_price.legend(loc="upper left", frameon=True, facecolor="#0d1117", edgecolor="#30363d", fontsize=8.5, ncol=2)
ax_price.tick_params(labelbottom=False)

# --- Panel 2: Order Flow Volume Profile ---
bar_height = (bins[1] - bins[0]) * 0.82
ax_vp.barh(bin_centers, profile["Buy_Vol"], height=bar_height, color="#2ea043", alpha=0.85, label="Buy Vol")
ax_vp.barh(bin_centers, -profile["Sell_Vol"], height=bar_height, color="#da3633", alpha=0.85, label="Sell Vol")
ax_vp.axvline(0, color="#8b949e", linewidth=0.8)
ax_vp.axhline(poc_price, color="#f85149", linestyle=":", linewidth=1.3)
ax_vp.set_title("Volume Profile", fontsize=11, fontweight="bold", color="#f0f6fc", pad=12)
ax_vp.set_xlabel("Sell  |  Buy", color="#c9d1d9", fontsize=9)
ax_vp.tick_params(labelleft=False, labelsize=8)
ax_vp.legend(loc="lower right", frameon=True, facecolor="#0d1117", edgecolor="#30363d", fontsize=8)

# --- Panel 3: Accumulation / Distribution Line ---
ax_ad.plot(data.index, data["AD_Line"], color="#3fb950", linewidth=1.6, label="Accumulation / Distribution (Smart Money Flow)")
ax_ad.fill_between(data.index, data["AD_Line"], data["AD_Line"].min(), color="#3fb950", alpha=0.12)
ax_ad.set_ylabel("A/D Flow", color="#c9d1d9", fontweight="bold")
ax_ad.set_xlabel("Date", color="#c9d1d9", fontweight="bold")
ax_ad.legend(loc="upper left", frameon=True, facecolor="#0d1117", edgecolor="#30363d", fontsize=9)

plt.tight_layout()
plt.show()
