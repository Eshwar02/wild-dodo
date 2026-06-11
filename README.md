# AlphaLearn — RL stock trading demo

Small RL project that trains a PPO agent to buy/hold/sell a single asset (e.g. SPY) on historical daily data.

Features
- Downloads and caches OHLCV via yfinance (data_cache/)
- Computes simple indicators: ret1, SMA ratio, vol10, RSI
- Chronological train/test split (test period is quarantined)
- Gymnasium env (TradingEnv) with discrete actions: hold, buy, sell
- PPO training (stable-baselines3)
- Deterministic evaluation vs buy-and-hold with return/Sharpe/max-drawdown
- Streamlit demo (app.py) with live training and test-set visualization

Quick start
1. Create a venv and install deps (example):

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Train from CLI:
   python train.py --ticker SPY --steps 200000

3. Run demo UI (Streamlit):
   .venv/bin/streamlit run app.py

Notes
- The test split is deliberately held out for integrity — do not tune on it.
- This project is for research/demonstration only (not financial advice).

Files
- data.py — data download, indicators, split
- env.py — TradingEnv gymnasium environment
- train.py — offline SB3 PPO training
- evaluate.py — run agent on test split and compute metrics
- app.py — Streamlit demo with live training and post-train evaluation

Saved models are in models/ (untrained + trained checkpoints for SPY).

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
