# Modern Portfolio Theory

## Origins
Modern Portfolio Theory (MPT) was developed by Harry Markowitz in 1952 and earned him the Nobel Prize in Economics. It provides a mathematical framework for constructing portfolios that maximize expected return for a given level of risk.

## Core Insight
The risk of a portfolio is not the average risk of its components — it depends on how the assets move relative to each other (correlation). Combining assets with low or negative correlation reduces overall portfolio risk without proportionally reducing return.

## Key Concepts

### Expected Return
Weighted average of expected returns of each asset in the portfolio.

### Portfolio Variance (Risk)
Depends on:
- Each asset's individual variance
- The weight of each asset
- The covariance (correlation) between each pair of assets

Adding an asset with a low correlation to an existing portfolio can reduce total risk even if the new asset is risky on its own.

### Correlation Coefficient
Ranges from -1 to +1:
- **+1**: Assets move in perfect lockstep. No diversification benefit.
- **0**: No relationship. Some diversification benefit.
- **-1**: Assets move perfectly opposite. Maximum diversification benefit.

## The Efficient Frontier
Plotting all possible portfolios of risky assets creates a curve. The top edge is the "efficient frontier" — portfolios that offer the highest return for each level of risk.

Any portfolio on the efficient frontier is "efficient" — you can't get more return without more risk, or less risk without less return.

## Capital Market Line
Adding a risk-free asset (T-bills) to the analysis creates the Capital Market Line. The tangent point between the Capital Market Line and the efficient frontier is the "market portfolio" — the theoretically optimal risky portfolio. This is essentially the case for total market index funds.

## Sharpe Ratio
Measures risk-adjusted return: (Portfolio return - Risk-free rate) / Portfolio standard deviation.

The portfolio on the Capital Market Line with the highest Sharpe ratio is the market portfolio.

## Implications for Individual Investors
1. **Diversification is the only free lunch in investing** — you can reduce risk without sacrificing expected return.
2. **The market portfolio is hard to beat** — supports passive index investing.
3. **Every investor's optimal portfolio is on the Capital Market Line** — a combination of the market portfolio and the risk-free asset, adjusted by risk tolerance.

## Criticisms and Limitations
- Assumes normally distributed returns (real returns have fat tails — rare crashes are more common than the model predicts).
- Assumes rational investors — behavioral finance shows we're not.
- Relies on historical correlations that can break down in crises (stocks and bonds fell together in 2022).
- Mean-variance optimization is highly sensitive to input assumptions (garbage in, garbage out).

## Post-MPT Developments
- **CAPM (Capital Asset Pricing Model)**: Single-factor model where beta (market exposure) explains return.
- **Fama-French Three-Factor Model**: Adds size and value factors to beta.
- **Black-Litterman Model**: Combines MPT with investor views.
- **Risk Parity**: Weight assets by risk contribution, not dollar amount.