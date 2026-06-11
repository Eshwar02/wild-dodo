# AlphaLearn — RL stock trading demo

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An RL project that trains a PPO agent to **buy / hold / sell** a single asset on real
historical daily data, then evaluates it on a held-out future period against a
buy-and-hold baseline. Includes an interactive Streamlit UI for training and visualizing
the agent.

> Research / demonstration only. The agent trades a **simulated** portfolio on daily
> bars — not a live broker. Not financial advice.

## What it does

- Downloads and caches real OHLCV via `yfinance` (`data_cache/`), fetched **through today**
  with automatic refresh of stale caches.
- Computes simple indicators: 1-day return, SMA(10/30) ratio, 10-day volatility, RSI(14).
- **Chronological** train/test split (train ≤ 2022-12-31, test = 2023 → today). The test
  period is quarantined — never used for training or tuning.
- `TradingEnv` (Gymnasium) with discrete actions: `0=hold, 1=buy/all-in, 2=sell/all-cash`,
  reward = daily portfolio P&L minus a transaction cost.
- PPO training via `stable-baselines3` (PyTorch, CPU).
- Deterministic evaluation vs buy-and-hold: return, Sharpe, max drawdown.
- Streamlit UI (`app.py`): live training reward curve, before/after toggle (untrained vs
  trained), trade markers on the price chart, equity curve, and animated day-by-day playback.

## How much is the agent trained?

Training is measured in **timesteps** — each timestep is one decision the agent makes
(one simulated trading day). PPO collects experience in rollouts of `n_steps=2048` and
updates the network with `batch_size=256`. The training data is ~3,200 daily bars (SPY,
2010–2022), so the figures below are roughly "how many times the agent trades through
history."

| Setting (UI slider) | Timesteps | ≈ passes over train data | Use |
|---------------------|-----------|--------------------------|-----|
| Quick smoke         | 5,000     | ~1.5                     | Verify the pipeline runs |
| Light               | 20,000    | ~6                       | Rough signal |
| Default             | 50,000    | ~15                      | Reasonable demo |
| Strong              | 100,000   | ~30                      | Better-converged agent |
| Full                | 200,000   | ~60                      | Best results (CLI default) |

- **UI default:** 50,000 timesteps (`app.py` slider).
- **CLI default:** 200,000 timesteps (`train.py --steps`).
- More timesteps = more learning, with diminishing returns; watch the live reward curve
  to see when it plateaus.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train from the CLI:

```bash
python train.py --ticker SPY --steps 200000
```

Evaluate on the held-out test period:

```bash
python evaluate.py --ticker SPY
```

Run the interactive UI:

```bash
.venv/bin/streamlit run app.py
```

## Tickers

- US stocks/ETFs: plain symbol — `SPY`, `AAPL`, `MSFT`, `TSLA`.
- **Indian stocks need a suffix**: `.NS` for NSE, `.BO` for BSE — e.g. `RELIANCE.NS`,
  `ITC.NS`, `INFY.NS`. Plain `RELIANCE` will not work.
- A ticker must have history on both sides of the 2022 train/test split, or the app shows
  a clear "not enough data" message.

## How to tell the agent learned

1. **Reward curve climbs** during training (learning happened — but could be memorization).
2. **Beats buy-and-hold on the unseen test period** (higher return, or similar return with
   lower drawdown / higher Sharpe) — the real evidence.
3. **Before/after toggle**: trained agent clearly beats the untrained (random) one.
4. **Holds up across several tickers**, not just one lucky pick.

Note: this is a trading *policy* (it chooses actions), not a price *predictor* — it never
outputs a next price. It is judged by simulated P&L on unseen data.

## Files

| File | Purpose |
|------|---------|
| `data.py` | Data download (through today), indicators, chronological split |
| `env.py` | `TradingEnv` Gymnasium environment |
| `train.py` | CLI PPO training; saves trained + untrained checkpoints |
| `evaluate.py` | Run agent on the test split, compute metrics vs buy-and-hold |
| `app.py` | Streamlit UI: live training + evaluation visuals |
| `PRD_rl_stock_trader.md` | Product requirements / design doc |

Saved models go to `models/` (git-ignored). Cached data goes to `data_cache/` (git-ignored).
