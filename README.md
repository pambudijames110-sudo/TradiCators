# Quantitative Market Microstructure & Order Flow Dashboard

A Python-based financial analytics dashboard that combines **Price Action**, **Volume-by-Price Order Flow Profiles**, **Accumulation/Distribution (A/D) Flow**, and **Wyckoff Volume Spread Analysis (VSA)** to distinguish between institutional positioning and retail panic selling.

## Key Features
* **Automated Regime Classification**: Evaluates trailing 10-day volume Z-scores and candle close locations to classify the current market state (*Institutional Accumulation*, *Institutional Distribution*, *Retail Capitulation*, or *Neutral*).
* **Smart Money vs. Retail Signals**:
  * **Institutional Accumulation (`^`)**: High statistical volume (`Z-Score > 1.2`) with closes in the upper range (`CLV >= 0.6`).
  * **Institutional Outflow (`v`)**: Heavy selling volume (`Z-Score > 1.2`) closing near the session lows (`CLV <= -0.6`).
  * **Retail Panic / Absorption (`◆`)**: Extreme volume spikes (`Z-Score > 2.0`) on down days where price rebounds off the lows (`CLV > -0.3`), signaling institutional absorption.
* **Volume Profile & Point of Control (POC)**: Maps buying vs. selling volume across 25 price bins to identify high-liquidity support/resistance zones.
* **Cumulative Accumulation/Distribution Line**: Tracks money flow divergence relative to price trends.

## Tech Stack
* **Python 3.10+**
* **Data Retrieval**: `yfinance`
* **Quantitative Analysis**: `pandas`, `numpy`
* **Visualization**: `matplotlib` (Custom `GridSpec` dark-mode layout)

## Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com
   cd market-microstructure-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install yfinance pandas numpy matplotlib
   ```

3. **Run the dashboard**
   ```bash
   python dashboard.py
   ```
   Enter any global or local ticker symbol when prompted (e.g., `NVDA`, `AAPL`, `BBCA.JK`) and select a lookback period (`1mo`, `3mo`, `6mo`, `1y`).

## Methodology Notes
* **Close Location Value (CLV)**:
  $$\text{CLV} = \frac{(\text{Close} - \text{Low}) - (\text{High} - \text{Close})}{\text{High} - \text{Low}}$$
* **Volume Z-Score**: Standardized against a 20-day rolling mean and standard deviation to normalize liquidity differences across large-cap and mid-cap equities.
