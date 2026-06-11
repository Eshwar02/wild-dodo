"""Interactive training UI for the RL stock-trading agent.

Run:  .venv/bin/streamlit run app.py

Three panels:
  1. Sidebar controls (ticker, steps, learning rate, txn cost).
  2. Live training — reward curve + progress update DURING model.learn().
  3. Post-train evaluation on the unseen test split, with trade markers
     and an equity curve vs buy-and-hold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from data import load
from env import TradingEnv, BUY, SELL
import evaluate as ev

st.set_page_config(page_title="AlphaLearn — RL Trading", layout="wide")


# --- live training callback -------------------------------------------------
class StreamlitCallback(BaseCallback):
    """Pushes rolling mean episode reward to Streamlit placeholders live."""

    def __init__(self, total_steps, prog_bar, status, chart_ph, update_every=2048):
        super().__init__()
        self.total_steps = total_steps
        self.prog_bar = prog_bar
        self.status = status
        self.chart_ph = chart_ph
        self.update_every = update_every
        self.xs, self.ys = [], []
        self._last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.update_every:
            self._last = self.num_timesteps
            buf = self.model.ep_info_buffer
            if buf:
                mean_r = float(np.mean([e["r"] for e in buf]))
                self.xs.append(self.num_timesteps)
                self.ys.append(mean_r)
                fig = go.Figure(go.Scatter(x=self.xs, y=self.ys, mode="lines+markers",
                                           line=dict(color="#00cc96", width=2)))
                fig.update_layout(title="Mean episode reward (live)",
                                  xaxis_title="timesteps", yaxis_title="reward ($ P&L)",
                                  height=320, margin=dict(l=10, r=10, t=40, b=10))
                self.chart_ph.plotly_chart(fig, use_container_width=True)
            frac = min(1.0, self.num_timesteps / self.total_steps)
            self.prog_bar.progress(frac)
            self.status.write(f"step {self.num_timesteps:,} / {self.total_steps:,}")
        return True


# --- cached data ------------------------------------------------------------
@st.cache_data(show_spinner="Loading price data…")
def get_data(ticker):
    return load(ticker)


# --- sidebar controls -------------------------------------------------------
st.sidebar.title("⚙️ Training controls")
ticker = st.sidebar.text_input("Ticker", value="SPY").upper().strip()
steps = st.sidebar.select_slider("Training steps",
                                 options=[5_000, 20_000, 50_000, 100_000, 200_000],
                                 value=50_000)
lr = st.sidebar.select_slider("Learning rate",
                              options=[1e-4, 3e-4, 1e-3], value=3e-4,
                              format_func=lambda x: f"{x:.0e}")
txn = st.sidebar.slider("Transaction cost", 0.0, 0.005, 0.001, 0.0005,
                        format="%.4f")
window = st.sidebar.slider("Observation window (days)", 3, 20, 5)
go_btn = st.sidebar.button("🚀 Train agent", type="primary", use_container_width=True)

st.title("📈 AlphaLearn — watch an RL agent learn to trade")
st.caption("Trains on past data, evaluated on a quarantined future test set. "
           "Not financial advice.")


# --- main flow --------------------------------------------------------------
try:
    train_df, test_df = get_data(ticker)
except Exception as e:
    st.error(f"Could not load data for '{ticker}': {e}")
    st.stop()

if len(train_df) <= window or len(test_df) <= window:
    st.error(
        f"'{ticker}' doesn't have enough history on both sides of the 2022 train/test "
        f"split (train={len(train_df)} rows, test={len(test_df)} rows, window={window}). "
        "Pick a ticker with a long history that trades through 2023–2024 (e.g. SPY, AAPL, MSFT)."
    )
    st.stop()

c1, c2 = st.columns(2)
c1.metric("Train period", f"{train_df.index.min().date()} → {train_df.index.max().date()}",
          f"{len(train_df)} days")
c2.metric("Test period (unseen)", f"{test_df.index.min().date()} → {test_df.index.max().date()}",
          f"{len(test_df)} days")

if go_btn:
    st.subheader("① Live training")
    prog = st.progress(0.0)
    status = st.empty()
    chart_ph = st.empty()

    venv = DummyVecEnv([lambda: Monitor(TradingEnv(train_df, window=window,
                                                   txn_cost=txn, random_start=True))])
    model = PPO("MlpPolicy", venv, learning_rate=lr, n_steps=2048, batch_size=256,
                gae_lambda=0.95, gamma=0.99, ent_coef=0.01, device="cpu", verbose=0)
    # snapshot the untrained policy = the "before" agent for the toggle
    import io
    snap = io.BytesIO()
    model.save(snap)
    snap.seek(0)
    st.session_state["untrained"] = PPO.load(snap, device="cpu")
    cb = StreamlitCallback(steps, prog, status, chart_ph)
    with st.spinner("Training…"):
        model.learn(total_timesteps=steps, callback=cb)
    prog.progress(1.0)
    status.success("Training complete.")
    st.session_state["model"] = model
    st.session_state["cfg"] = dict(window=window, txn=txn)

# --- rollout helper ---------------------------------------------------------
def rollout(model, df, cfg):
    """Run a policy deterministically over df. Returns values, actions, prices."""
    env = TradingEnv(df, window=cfg["window"], txn_cost=cfg["txn"], random_start=False)
    obs, _ = env.reset()
    values, actions, prices = [env._prev_value], [], []
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        a = int(a)
        obs, _, term, trunc, info = env.step(a)
        values.append(info["portfolio_value"])
        actions.append(a)
        prices.append(info["price"])
        done = term or trunc
    return np.array(values), actions, prices


# --- evaluation -------------------------------------------------------------
if "model" in st.session_state:
    st.subheader("② Evaluation on unseen test data")
    cfg = st.session_state["cfg"]

    # before/after toggle
    choice = st.radio("Which agent to evaluate?",
                      ["Trained (after)", "Untrained (before)"],
                      horizontal=True)
    model = (st.session_state["model"] if choice.startswith("Trained")
             else st.session_state["untrained"])

    agent_vals, actions, prices = rollout(model, test_df, cfg)
    bh_vals = ev.buy_and_hold(test_df)
    am, bm = ev.metrics(agent_vals), ev.metrics(bh_vals)

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{choice.split()[0]} return", f"{am['return']*100:.2f}%",
              f"{(am['return']-bm['return'])*100:+.2f}% vs B&H")
    m2.metric("Sharpe", f"{am['sharpe']:.2f}", f"B&H {bm['sharpe']:.2f}")
    m3.metric("Max drawdown", f"{am['max_drawdown']*100:.2f}%",
              f"B&H {bm['max_drawdown']*100:.2f}%")

    # quick before-vs-after summary so the toggle's value is obvious
    if "untrained" in st.session_state:
        uv, _, _ = rollout(st.session_state["untrained"], test_df, cfg)
        tv, _, _ = rollout(st.session_state["model"], test_df, cfg)
        st.caption(f"Before training: {ev.metrics(uv)['return']*100:+.2f}% return  →  "
                   f"After training: {ev.metrics(tv)['return']*100:+.2f}% return")

    dates = test_df.index[cfg["window"]-1:][:len(prices)]
    buys = [(d, p) for d, a, p in zip(dates, actions, prices) if a == BUY]
    sells = [(d, p) for d, a, p in zip(dates, actions, prices) if a == SELL]

    # static price chart with buy/sell markers
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=test_df.index, y=test_df["Close"],
                              mode="lines", name="price", line=dict(color="#888")))
    if buys:
        fig1.add_trace(go.Scatter(x=[d for d, _ in buys], y=[p for _, p in buys],
                                  mode="markers", name="BUY",
                                  marker=dict(color="#00cc96", size=10, symbol="triangle-up")))
    if sells:
        fig1.add_trace(go.Scatter(x=[d for d, _ in sells], y=[p for _, p in sells],
                                  mode="markers", name="SELL",
                                  marker=dict(color="#ef553b", size=10, symbol="triangle-down")))
    fig1.update_layout(title=f"{choice} — trades on unseen test data", height=380,
                       margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig1, use_container_width=True)

    # equity curve vs buy & hold
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=test_df.index[:len(agent_vals)], y=agent_vals, name=choice,
                              line=dict(color="#00cc96", width=2)))
    fig2.add_trace(go.Scatter(x=test_df.index[:len(bh_vals)], y=bh_vals, name="Buy & Hold",
                              line=dict(color="#636efa", width=2, dash="dash")))
    fig2.update_layout(title="Portfolio value vs buy-and-hold", height=380,
                       yaxis_title="$", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    # --- animated playback ---------------------------------------------------
    st.subheader("③ Animated playback — watch it trade day by day")
    n = len(prices)
    step = max(1, n // 120)  # cap frames for smooth playback
    pdates = list(dates)
    pa = actions

    base = go.Scatter(x=pdates[:1], y=prices[:1], mode="lines",
                      line=dict(color="#888"), name="price")
    frames = []
    for k in range(1, n + 1, step):
        bx = [pdates[i] for i in range(k) if pa[i] == BUY]
        by = [prices[i] for i in range(k) if pa[i] == BUY]
        sx = [pdates[i] for i in range(k) if pa[i] == SELL]
        sy = [prices[i] for i in range(k) if pa[i] == SELL]
        frames.append(go.Frame(data=[
            go.Scatter(x=pdates[:k], y=prices[:k], mode="lines", line=dict(color="#888")),
            go.Scatter(x=bx, y=by, mode="markers",
                       marker=dict(color="#00cc96", size=11, symbol="triangle-up")),
            go.Scatter(x=sx, y=sy, mode="markers",
                       marker=dict(color="#ef553b", size=11, symbol="triangle-down")),
        ], name=str(k)))

    anim = go.Figure(
        data=[base,
              go.Scatter(x=[], y=[], mode="markers", name="BUY",
                         marker=dict(color="#00cc96", size=11, symbol="triangle-up")),
              go.Scatter(x=[], y=[], mode="markers", name="SELL",
                         marker=dict(color="#ef553b", size=11, symbol="triangle-down"))],
        frames=frames,
    )
    anim.update_xaxes(range=[pdates[0], pdates[-1]])
    anim.update_yaxes(range=[min(prices) * 0.97, max(prices) * 1.03])
    anim.update_layout(
        title="Trade playback", height=420, margin=dict(l=10, r=10, t=40, b=10),
        updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=1.15,
                          buttons=[
                              dict(label="▶ Play", method="animate",
                                   args=[None, dict(frame=dict(duration=60, redraw=True),
                                                    fromcurrent=True)]),
                              dict(label="⏸ Pause", method="animate",
                                   args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                      mode="immediate")]),
                          ])],
    )
    st.plotly_chart(anim, use_container_width=True)
else:
    st.info("Set parameters in the sidebar and click **Train agent** to begin.")
