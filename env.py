"""TradingEnv — gymnasium env for single-asset long/cash trading.

Actions (Discrete 3): 0=hold, 1=buy/go long (all-in), 2=sell/go to cash.
Reward = change in portfolio value minus transaction cost on trade days.
Transaction cost is non-zero by design — without it the agent flickers and
the result is fiction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from data import FEATURE_COLS

HOLD, BUY, SELL = 0, 1, 2


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, window: int = 5,
                 txn_cost: float = 0.001, start_cash: float = 10_000.0,
                 random_start: bool = True):
        super().__init__()
        self.df = df.reset_index(drop=False).rename(columns={"index": "date"})
        self.prices = self.df["Close"].to_numpy(dtype=np.float64)
        self.feats = self.df[FEATURE_COLS].to_numpy(dtype=np.float32)
        self.window = window
        self.txn_cost = txn_cost
        self.start_cash = start_cash
        self.random_start = random_start

        n_feat = len(FEATURE_COLS) * window + 1  # +1 position flag
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(n_feat,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self._t = 0
        self._t_end = len(self.prices) - 1
        self.position = 0  # 0=cash, 1=long
        self.cash = start_cash
        self.shares = 0.0

    # --- helpers ---
    def _portfolio_value(self, price: float) -> float:
        return self.cash + self.shares * price

    def _obs(self) -> np.ndarray:
        w = self.feats[self._t - self.window + 1: self._t + 1].flatten()
        return np.concatenate([w, [np.float32(self.position)]]).astype(np.float32)

    # --- gym API ---
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        lo = self.window - 1
        hi = max(lo, self._t_end - 1)
        self._t = self.np_random.integers(lo, hi) if (self.random_start and hi > lo) else lo
        self.position = 0
        self.cash = self.start_cash
        self.shares = 0.0
        self._prev_value = self._portfolio_value(self.prices[self._t])
        return self._obs(), {}

    def step(self, action: int):
        price = self.prices[self._t]
        cost = 0.0

        if action == BUY and self.position == 0:
            self.shares = self.cash / price
            cost = self.cash * self.txn_cost
            self.cash = 0.0
            self.position = 1
        elif action == SELL and self.position == 1:
            proceeds = self.shares * price
            cost = proceeds * self.txn_cost
            self.cash = proceeds
            self.shares = 0.0
            self.position = 0

        # advance time, mark to market at next price
        self._t += 1
        next_price = self.prices[self._t]
        value = self._portfolio_value(next_price) - cost
        reward = value - self._prev_value
        self._prev_value = value

        terminated = False
        truncated = self._t >= self._t_end
        info = {"portfolio_value": value, "position": self.position, "price": next_price}
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        print(f"t={self._t} price={self.prices[self._t]:.2f} "
              f"pos={self.position} value={self._prev_value:.2f}")


if __name__ == "__main__":
    from data import load
    from gymnasium.utils.env_checker import check_env
    train, _ = load()
    env = TradingEnv(train, random_start=False)
    check_env(env)
    print("check_env OK  obs_dim:", env.observation_space.shape)
