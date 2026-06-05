# Options Basics

## What Is an Options Contract?
An options contract gives the buyer the **right, but not the obligation**, to buy or sell an underlying asset (usually a stock) at a specified price (strike price) before or on a specified date (expiration).

Options are derivatives — their value derives from an underlying asset.

## Two Types of Options

### Call Option
The right to **buy** 100 shares at the strike price.
- Buyer profits if stock rises above strike price + premium paid.
- Seller profits if stock stays below strike price.

### Put Option
The right to **sell** 100 shares at the strike price.
- Buyer profits if stock falls below strike price - premium paid.
- Seller profits if stock stays above strike price.

## Key Terms
- **Premium**: The price paid for the option. This is the maximum loss for a buyer.
- **Strike price**: The price at which the underlying can be bought/sold.
- **Expiration date**: Last day the option can be exercised.
- **In the money (ITM)**: Call: stock above strike. Put: stock below strike.
- **Out of the money (OTM)**: Call: stock below strike. Put: stock above strike.
- **At the money (ATM)**: Stock price equals strike price.

## Options Pricing: The Greeks
- **Delta**: How much the option price moves for a $1 move in the stock. Call: 0 to +1. Put: -1 to 0.
- **Theta**: Time decay. Options lose value as expiration approaches (all else equal). Works against buyers, for sellers.
- **Vega**: Sensitivity to volatility. Higher implied volatility = higher option price.
- **Gamma**: Rate of change of delta. Highest for ATM options near expiration.

## Basic Strategies

### Buying Calls (Bullish)
Profits if stock rises significantly. Limited loss (premium paid). Leveraged upside.

### Buying Puts (Bearish or Protection)
Profits if stock falls. Buying puts on existing positions = portfolio insurance.

### Covered Call (Income Generation)
Own 100 shares. Sell a call option against them. Collect premium income. Caps upside at strike price.
Popular for generating extra income on stocks you own.

### Cash-Secured Put (Stock Acquisition)
Sell a put option. Keep enough cash to buy shares if assigned. Collect premium. Buy stock at a discount if assigned.
Popular for entering stock positions at lower prices.

### Protective Put
Own shares + buy a put. Like insurance on your position. Costs premium but limits downside.

## Risks
- **Buyers**: Can lose 100% of premium if option expires worthless. This happens frequently.
- **Naked option sellers**: Theoretically unlimited loss on naked calls. Large loss potential on naked puts.
- **Time decay**: Options lose value as expiration approaches. Buyers fight this.
- **Complexity**: Greeks, expiration, assignment risk make options difficult to master.

## Options vs. Stocks
| Feature | Stock | Option |
|---------|-------|--------|
| Cost | Full share price | Premium only (leverage) |
| Expiration | None | Fixed date |
| Max loss | 100% (no leverage) | 100% of premium (buyer) |
| Complexity | Low | High |
| Income generation | Dividends | Premium selling |

## Appropriate Use
Options are appropriate for:
- Portfolio hedging (buying protective puts).
- Generating income on existing positions (covered calls).
- Speculating with limited capital (buying calls/puts — but most expire worthless).

Options are **not** appropriate as a primary investment vehicle for most retail investors. Most people who trade options lose money net of fees and spreads.