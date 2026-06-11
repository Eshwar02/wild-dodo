"""Evaluate trained agent on the QUARANTINED test split vs buy-and-hold.

Integrity centerpiece: agent runs deterministically on data it never trained on.
Reports return, Sharpe, max drawdown for agent and baseline.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from stable_baselines3 import PPO

from data import load
from env import TradingEnv

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def metrics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    rets = np.diff(values) / values[:-1]
    total_return = values[-1] / values[0] - 1.0
    sharpe = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252) if len(rets) else 0.0
    peak = np.maximum.accumulate(values)
    max_dd = ((values - peak) / peak).min()
    return {"return": total_return, "sharpe": sharpe, "max_drawdown": max_dd}


def run_agent(model, df) -> np.ndarray:
    env = TradingEnv(df, random_start=False)
    obs, _ = env.reset()
    values = [env._prev_value]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(action))
        values.append(info["portfolio_value"])
        done = term or trunc
    return np.array(values)


def buy_and_hold(df, start_cash=10_000.0) -> np.ndarray:
    prices = df["Close"].to_numpy(dtype=np.float64)
    shares = start_cash / prices[0]
    return shares * prices


def fmt(m: dict) -> str:
    return f"return={m['return']*100:6.2f}%  sharpe={m['sharpe']:5.2f}  maxDD={m['max_drawdown']*100:6.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    args = ap.parse_args()

    _, test_df = load(args.ticker)
    model = PPO.load(os.path.join(MODEL_DIR, f"{args.ticker}_trained"))

    agent_vals = run_agent(model, test_df)
    bh_vals = buy_and_hold(test_df)

    am, bm = metrics(agent_vals), metrics(bh_vals)
    print(f"\nTEST period: {test_df.index.min().date()} .. {test_df.index.max().date()}  (UNSEEN)")
    print(f"  Agent       : {fmt(am)}")
    print(f"  Buy & Hold  : {fmt(bm)}")
    win = "AGENT WINS" if am["return"] > bm["return"] else "baseline wins on return"
    print(f"  -> {win} (return).  Lower drawdown is also a win.\n")


if __name__ == "__main__":
    main()
