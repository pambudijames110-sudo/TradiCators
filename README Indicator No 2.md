# TradiCators — Quantitative Market Microstructure & Statistical Arbitrage Suite

A Python quantitative finance repository featuring two institutional-style analytics engines:
1. **Market Microstructure & Order Flow Dashboard (`Indicator no 1.py`)**: Detects institutional accumulation, distribution, and retail panic selling using Wyckoff Volume Spread Analysis (VSA) and Volume Profiles.
2. **Walk-Forward Multi-Factor Statistical Arbitrage (`Indicator no 2.py`)**: Constructs a Dollar-Neutral, Beta-Neutral Long/Short portfolio using Ridge Regression ($L_2$) and analytical null-space projection.

---

## Project 1: Market Microstructure & Order Flow Dashboard (`Indicator no 1.py`)

![Market Microstructure Preview](preview.png)

### Key Methodology
* **Automated Regime Classification**: Evaluates trailing 10-day volume Z-scores and Close Location Value ($\text{CLV}$) to classify market states (*Institutional Accumulation*, *Institutional Distribution*, *Retail Capitulation*, or *Neutral*).
* **Volume-by-Price Profile & Point of Control (POC)**: Maps buying vs. selling pressure across 25 price bins to identify high-liquidity support and resistance levels.
* **Cumulative Accumulation/Distribution (A/D) Flow**: Tracks smart-money divergence relative to price action.

---

## Project 2: Walk-Forward Multi-Factor Statistical Arbitrage (`Indicator no 2.py`)

![Quantitative Tear Sheet](tearsheet_preview.png)

### Key Methodology
* **Multi-Factor Ridge Regression ($L_2$ Regularization)**: Estimates asset sensitivities to macro factors (`^JKSE`, `IDR=X`, `GC=F`) via $\hat{\beta} = (X^T X + \lambda I)^{-1} X^T Y$ with an unpenalized $\alpha$ intercept.
* **Look-Ahead Bias Prevention**: Lags non-Jakarta macro factors by 1 trading day (`shift(1)`) so Indonesian market trades only condition on prior information.
* **Closed-Form Dollar- & Beta-Neutral Solver**: Uses exact orthogonal null-space projection to enforce $\sum w_i = 0$ and $w^T \beta_{\text{market}} = 0$ (achieving near-zero realized market correlation `0.04`).
* **Monthly Rolling Walk-Forward Validation**: Re-estimates betas every 21 trading days over a 252-day trailing window with realistic transaction cost drag (`20 bps`).

---

## Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com
cd TradiCators

# 2. Install dependencies
pip install yfinance numpy pandas matplotlib

# 3. Run either engine
python "Indicator no 1.py"
python "Indicator no 2.py"
```

## Real-World Execution Caveats (IDX Market Structure)
* Retail short-selling on the Indonesia Stock Exchange (BEI/IDX) is restricted to margin-approved LQ45 securities. In live production, the short leg of a market-neutral Indonesian equity book is typically hedged via **LQ45 index futures** or **long-only benchmark-relative tilts**.

