# PRD — AlphaLearn: An RL Stock-Trading Agent

**Project type:** Reinforcement learning environment + trained trading agent + live visualizer
**Context:** AI hackathon, 4-day build, team of 2–3, goal = impress judges with an interactive demo
**Status:** Draft v1
**Last updated:** 2026-06-11

---

## 1. One-line pitch

> An RL agent learns *when to buy, hold, and sell* a stock from historical price data — and we prove it beats buy-and-hold on data it has never seen.

---

## 2. Problem & motivation

Trading is a sequential decision problem under uncertainty — a natural fit for RL. Instead of predicting price (hard, noisy), the agent learns a *policy*: given the recent market state, what action maximizes long-term return? The agent discovers timing strategies on its own from raw price/indicator data.

**Why this works for a hackathon:**
- **Instantly relatable.** Everyone understands "did it make money?"
- **Single killer metric.** Portfolio value vs. buy-and-hold on a held-out test period.
- **Rich visuals.** Buy/sell markers on a price chart + an equity curve climbing.

**The two traps that kill RL-trading demos (and how we avoid them):**
1. **Data leakage / lookahead bias** → fake spectacular returns. *Mitigation:* strict chronological train/test split; the agent is evaluated ONLY on a test period it never trained on. We say this out loud in the pitch — judges respect rigor.
2. **"A line goes up" is boring.** *Mitigation:* invest in the visualizer — animated buy/sell markers, side-by-side equity curves, and a live "watch it trade the test set" playback.

**Non-goals (explicitly out of scope):**
- Live brokerage integration or real money.
- High-frequency / order-book microstructure.
- Beating professional quant funds. We beat **buy-and-hold**, honestly, on out-of-sample data.
- Claiming this is investment advice (put a disclaimer in the demo).

---

## 3. Target users / audience

- **Primary:** Hackathon judges — want a clear "it learned to trade and it's not cheating" story.
- **Secondary:** The team — needs scope tight enough to finish in 4 days.

---

## 4. Success criteria

| Tier | Definition |
|------|------------|
| **Must-have (the 80% product)** | Custom Gymnasium trading env; PPO agent trains on a train period; evaluated on a held-out test period; compared against buy-and-hold; visualizer shows trades + equity curve. |
| **Should-have** | Polished web UI, animated trade playback on test data, Sharpe ratio + return + max-drawdown metrics, multiple test stocks. |
| **Nice-to-have (stretch)** | Multi-asset portfolio allocation; transaction-cost sensitivity slider; compare against a moving-average-crossover baseline too. |

**Hard metric for demo:** On the **held-out test period**, the agent's total return **beats buy-and-hold** *and* has a better risk-adjusted return (Sharpe). Honesty bar: if it ties buy-and-hold with lower drawdown, that still counts as a win — say so.

---

## 5. The RL environment specification

Built as a `gymnasium.Env` subclass. This is the core deliverable.

### 5.1 Data
- One liquid stock or ETF (e.g. SPY, AAPL) daily OHLCV via `yfinance`. Download once, cache to CSV (don't hit the API live during the demo).
- **Chronological split (critical):** e.g. 2015–2022 = train, 2023–2024 = test. Never shuffle. The test period is sacred — no peeking.
- Precompute a few technical indicators offline: returns, SMA, RSI, volatility. Keep it to ~5 features.

### 5.2 Observation space
`Box` vector, normalized, built from a rolling window of recent days:
- Recent normalized returns (last N days)
- A couple of indicators (e.g. RSI, SMA-ratio, rolling volatility)
- Current position flag (in market / in cash) — **the agent must know what it holds**
- (Optional) unrealized P&L of current position

→ Keep it ~10–20 dims. Small obs = faster, more stable training.

### 5.3 Action space
Start with the simplest that demos well — **`Discrete(3)`**:
- `0` = hold
- `1` = buy / go long (all-in)
- `2` = sell / go to cash

(Discrete is far easier to train than continuous position sizing. Continuous allocation is a stretch goal.)

### 5.4 Reward function
Reward each step = **change in portfolio value**, minus costs:

```
reward = (portfolio_value_t - portfolio_value_{t-1})        # daily P&L
       - transaction_cost   if a trade happened this step   # realism + anti-flicker
```

- Include a small transaction cost (e.g. 0.1%) from day one — without it the agent flickers in/out every step and the result is fake.
- Optionally use **log-returns** or a differential Sharpe ratio as reward for better risk behavior (stretch).
- Reward shaping is the biggest lever on demo quality — budget tuning time (Day 2–3).

### 5.5 Episode
- One episode = one full pass over the train period (or random sub-windows of it for variety).
- `reset()` returns to start (or a random start day in train), portfolio = starting cash, position = flat.

### 5.6 Baseline
**Buy-and-hold**: buy on day 1 of the test period, hold to the end. This is the bar the agent must beat. Build it Day 1 — it's trivial and it's your whole comparison story.

---

## 6. Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Env API | **Gymnasium** | Standard, SB3-compatible |
| RL algo | **Stable-Baselines3 PPO** | Proven; no custom training loop to debug — critical given "some RL exposure" |
| Data | **yfinance** + cached CSV | Free historical OHLCV; cache so the demo is offline-safe |
| Indicators | pandas / `ta` library | Precompute offline |
| Language | Python 3.10+ | Ecosystem |
| Demo frontend | **Web (FastAPI + Plotly/JS chart)** or Streamlit | Streamlit is fastest for a data demo; Plotly gives interactive charts |
| Checkpoints | SB3 `.zip` saves | Save trained agent + record test-set trades |

**De-risking principle:** We write the *environment, the data pipeline, and the demo* — not the RL algorithm. SB3 handles PPO internals. **Streamlit is a strong frontend choice here** — it's built for exactly this kind of data/chart demo and saves you a day vs. hand-rolling a web app.

---

## 7. Architecture

```
┌──────────────┐   ┌─────────────────┐   ┌──────────────────┐
│ yfinance →   │   │  TradingEnv     │   │  SB3 PPO trainer  │
│ cached CSV + │──▶│ (gymnasium.Env) │◄──│  (train.py)       │
│ indicators   │   │  step/reset/    │   └────────┬──────────┘
└──────────────┘   │  reward         │            │ saves
                   └────────┬────────┘            ▼
                            │            ┌──────────────────┐
                  TRAIN split            │  policy.zip      │
                  TEST split (held out)  └────────┬─────────┘
                            │                      │ load + run on TEST
                            ▼                      ▼
              ┌─────────────────────────────────────────────────┐
              │  Demo (Streamlit / web)                          │
              │  - price chart w/ buy/sell markers (test period) │
              │  - equity curve: agent vs buy-and-hold           │
              │  - metrics: return, Sharpe, max drawdown         │
              │  - "watch it trade" playback + disclaimer        │
              └─────────────────────────────────────────────────┘
```

---

## 8. Team split (2–3 people)

| Owner | Responsibility |
|-------|----------------|
| **A — Data + Environment** | yfinance download + caching, indicator precompute, chronological split, `TradingEnv` (spaces/step/reset/reward), buy-and-hold baseline. Done EOD Day 1. |
| **B — Training + Eval** | SB3 PPO setup, reward shaping, checkpointing, **the evaluation harness that runs the trained agent on the untouched test set** and computes return/Sharpe/drawdown. |
| **C — Demo/Frontend** (or shared) | Streamlit/web app: price chart with trade markers, dual equity curve, metrics panel, playback, pitch deck. |

---

## 9. 4-day timeline

### Day 1 — Data + environment foundations
- [ ] Repo + `pip install gymnasium stable-baselines3 yfinance pandas ta streamlit`
- [ ] Download + cache OHLCV; precompute indicators
- [ ] Chronological train/test split (lock the test period, never touch it in training)
- [ ] Implement `TradingEnv` (obs/action/reward, transaction costs)
- [ ] Implement buy-and-hold baseline + metric functions (return, Sharpe, drawdown)
- [ ] **Milestone:** env passes SB3 `check_env()`; random agent runs an episode

### Day 2 — Learning signal
- [ ] Wire SB3 PPO `model.learn()`
- [ ] Reward visibly increasing during training
- [ ] Build the **test-set evaluation harness** (this is the integrity centerpiece)
- [ ] First reward-shaping pass
- [ ] **Milestone:** trained agent beats buy-and-hold on the TEST split (any margin)
- [ ] **SCOPE FREEZE at end of Day 2**

### Day 3 — Demo build + polish
- [ ] Streamlit app: price chart + buy/sell markers, dual equity curve, metrics panel
- [ ] Longer training run for the final agent
- [ ] Tune reward/costs until the test-set win is solid and trades look sensible
- [ ] Add 2–3 test stocks so it's not a one-stock fluke

### Day 4 — Ship
- [ ] Lock the best checkpoint; pre-render the test-set trade log
- [ ] Rehearse the 2-minute pitch (problem → "trained here, tested HERE on unseen data" → beats buy-and-hold → it learned this itself)
- [ ] Record a backup demo video; add the "not financial advice" disclaimer
- [ ] **Milestone:** demo runs end-to-end twice without intervention

---

## 10. Risks & mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| **Data leakage → fake returns** | High | Strict chronological split; eval only on untouched test set; surface this in the pitch as a feature |
| Agent overfits train, fails test | Medium | Small obs space; transaction costs; multiple test stocks; report honest test results |
| Reward flickering (trades every day) | Medium | Transaction cost in reward from Day 1 |
| PPO doesn't beat buy-and-hold | Medium | Bar is buy-and-hold, not perfection; "ties with lower drawdown" still counts; reward shaping time boxed |
| Frontend overruns | Medium | Streamlit instead of custom web; record backup video |
| Scope creep (multi-asset, continuous sizing) | High | Stretch goals only; honor Day-2 freeze |
| yfinance API flaky during demo | Medium | Cache all data to CSV; never call the API live |

---

## 11. Demo script (2 minutes)

1. **Hook (15s):** "Can an AI learn to trade — not by being told the rules, but by trial and error? And can we trust the result?"
2. **The setup (20s):** "We trained it on 2015–2022. It has *never seen* 2023–2024. Watch it trade that unseen period now." (Honesty = credibility.)
3. **Watch it trade (40s):** Playback on the test chart — buy/sell markers appear; equity curve climbs alongside the buy-and-hold line.
4. **The result (30s):** Show final return, Sharpe, max drawdown vs. buy-and-hold. "It beat the benchmark on data it never saw — and it learned the strategy itself."
5. **Close (15s):** Stretch vision — multi-asset portfolios, risk-adjusted reward. *"Not financial advice."*

---

## 12. Honesty & integrity notes (don't skip — judges probe this)

- **Never** report train-set performance as if it were live performance.
- Keep the test period quarantined: no indicator normalization stats, no hyperparameter selection based on test results.
- Include realistic transaction costs; a frictionless agent's returns are fiction.
- State assumptions openly in the demo. A modest, *honest* out-of-sample win beats a suspicious 900% return every time.

---

## 13. Swapping the asset / variant

The framework holds for any tradable series — only Section 5's data changes. Variants with the same "relatable + single metric + survives a mediocre agent" properties:
- **Crypto** (BTC/ETH) — more volatility, more dramatic charts (also riskier/noisier).
- **Multi-stock portfolio allocation** — continuous action space (harder; stretch).
- **Pairs trading / mean reversion** — agent learns a spread strategy (more niche).

Keep the observation space small, start with discrete actions, enforce the chronological split, and always compare against buy-and-hold — those are the properties that make it shippable *and* credible in 4 days.
