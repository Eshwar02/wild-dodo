"""Train PPO agent on the TRAIN split. Saves trained + untrained checkpoints.

Untrained checkpoint = the "before" agent for the demo before/after toggle.
"""
from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from data import load
from env import TradingEnv

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def make_env(df):
    return lambda: TradingEnv(df, random_start=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--steps", type=int, default=200_000)
    args = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    train_df, _ = load(args.ticker)
    venv = DummyVecEnv([make_env(train_df)])

    model = PPO("MlpPolicy", venv, verbose=1, n_steps=2048, batch_size=256,
                gae_lambda=0.95, gamma=0.99, ent_coef=0.01, device="cpu", tensorboard_log=None)
    # save untrained "before" agent first
    model.save(os.path.join(MODEL_DIR, f"{args.ticker}_untrained"))
    model.learn(total_timesteps=args.steps)
    model.save(os.path.join(MODEL_DIR, f"{args.ticker}_trained"))
    print(f"saved: models/{args.ticker}_trained.zip  +  _untrained.zip")


if __name__ == "__main__":
    main()
