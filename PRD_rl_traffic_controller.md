# PRD — FlowState: An RL Traffic-Signal Controller

**Project type:** Reinforcement learning environment + trained agent + live visualizer
**Context:** AI hackathon, 4-day build, team of 2–3, goal = impress judges with an interactive demo
**Status:** Draft v1
**Last updated:** 2026-06-11

---

## 1. One-line pitch

> An RL agent learns to control a city intersection's traffic lights, turning gridlock into smooth flow — and you can watch it learn, live.

---

## 2. Problem & motivation

Fixed-timer traffic lights are dumb: they cycle on a schedule regardless of actual traffic. Real congestion is dynamic. This is a textbook RL problem — an agent observing queue lengths and choosing light phases can outperform fixed timers.

**Why this wins a hackathon:**
- **Visual & relatable.** Everyone understands traffic. The before/after (cars piling up vs. flowing) reads instantly from across the room.
- **Demo survives a mediocre agent.** Even a partially-trained policy beats a fixed timer, so the demo looks good even if training under-converges.
- **Clear metric.** "Average wait time reduced by X%" is a single number judges remember.

**Non-goals (explicitly out of scope):**
- Real-world deployment, real map data, or SUMO-grade traffic physics.
- Multi-intersection coordination (stretch goal only).
- Novel RL algorithm research — we use proven PPO, not a new method.

---

## 3. Target users / audience

- **Primary:** Hackathon judges — need a 2-minute "wow" and a clear metric.
- **Secondary:** The team — needs a scope tight enough to finish.

---

## 4. Success criteria

| Tier | Definition |
|------|------------|
| **Must-have (the 80% product)** | A custom Gymnasium env runs; PPO agent trains and measurably beats a fixed-timer baseline; live visualizer shows cars moving + a reward curve; before/after toggle works. |
| **Should-have** | Polished web UI, "watch it play" button, real-time wait-time counter, reward-shaping tuned so improvement is dramatic. |
| **Nice-to-have (stretch)** | Two coupled intersections; adjustable traffic-density slider; comparison view (fixed timer vs. agent side-by-side). |

**Hard metric for demo:** Agent reduces average vehicle wait time by **≥30%** vs. the fixed-timer baseline.

---

## 5. The RL environment specification

Built as a `gymnasium.Env` subclass. This is the core deliverable.

### 5.1 World model
- A single 4-way intersection (N, S, E, W approaches).
- Each approach has a queue of cars (discrete count, capped at e.g. 20).
- Cars arrive stochastically each step (Poisson-ish arrival rate per direction).
- A green light on an axis drains cars from that axis's queues at a fixed rate per step.

### 5.2 Observation space
`Box` vector, normalized to [0,1]:
- Queue length per approach (4 values)
- Current light phase (one-hot, e.g. NS-green vs EW-green) (2 values)
- Time since last phase change (1 value)

→ ~7-dim observation. Keep it small; small obs = fast, stable training.

### 5.3 Action space
`Discrete(2)`:
- `0` = set/keep NS green (EW red)
- `1` = set/keep EW green (NS red)

(Includes an implicit yellow/clearance penalty when switching — see reward.)

### 5.4 Reward function
Reward each step = **negative total waiting cars** (the agent minimizes accumulated queue):

```
reward = -(sum of all queue lengths)
       - switch_penalty if phase changed this step   # discourages flicker
```

Reward shaping is the single biggest lever on demo quality — budget tuning time for it (see Day 2–3).

### 5.5 Episode
- Fixed horizon (e.g. 500 steps).
- `reset()` clears queues, randomizes initial state and arrival rates.

### 5.6 Baseline
A fixed-timer policy (switch every N steps) for the before/after comparison. This is the thing the agent must beat — build it Day 1.

---

## 6. Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Env API | **Gymnasium** | Standard, SB3-compatible |
| RL algo | **Stable-Baselines3 PPO** | Proven, no custom training loop to debug — critical given "some RL exposure" |
| Language | Python 3.10+ | Ecosystem |
| Training viz | TensorBoard or live matplotlib | Reward curve |
| Demo frontend | **Web (FastAPI + simple JS canvas)** OR Pygame | Web reads better on a projector; Pygame is faster to build |
| Checkpoints | SB3 `.zip` saves | Save untrained + trained agents for before/after |

**De-risking principle:** We write the *environment and the demo*, not the RL algorithm. SB3 handles PPO internals.

---

## 7. Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  TrafficEnv     │◄────│  SB3 PPO trainer  │────▶│  saved policy.zip│
│ (gymnasium.Env) │     │  (train.py)       │     │  (trained +      │
│  step/reset/    │     └──────────────────┘     │   baseline)      │
│  render         │                                └────────┬─────────┘
└────────┬────────┘                                         │
         │ state stream                                      │ load
         ▼                                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Demo server / visualizer                                          │
│  - renders cars + lights    - reward/wait-time counter             │
│  - before/after toggle      - "watch it play" button               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Team split (2–3 people)

| Owner | Responsibility |
|-------|----------------|
| **A — Environment** | `TrafficEnv` (spaces, step, reset, reward), fixed-timer baseline, unit tests. Done EOD Day 1. |
| **B — Training** | SB3 PPO setup, reward-shaping iteration, checkpointing, logging reward curves. |
| **C — Demo/Frontend** (or shared between A/B) | Visualizer, before/after toggle, wait-time counter, pitch deck. |

---

## 9. 4-day timeline

### Day 1 — Foundations
- [ ] Repo + env setup, `pip install gymnasium stable-baselines3`
- [ ] Implement `TrafficEnv` with full step/reset/observation/reward
- [ ] Implement fixed-timer baseline policy
- [ ] Random-agent rollout renders/prints correctly
- [ ] **Milestone:** env passes `check_env()` from SB3

### Day 2 — Learning signal
- [ ] Wire SB3 PPO `model.learn()`
- [ ] Get reward visibly increasing over training
- [ ] First reward-shaping pass
- [ ] **Milestone:** trained agent beats baseline on wait time (any margin)
- [ ] **SCOPE FREEZE at end of Day 2** — no new features after this

### Day 3 — Demo build + polish
- [ ] Build the visualizer (cars, lights, counters)
- [ ] Before/after toggle using saved checkpoints
- [ ] Longer training run for the final agent
- [ ] Tune reward shaping until improvement is *dramatic* (target ≥30%)

### Day 4 — Ship
- [ ] Lock the best checkpoint
- [ ] Rehearse 2-minute pitch (problem → live before/after → metric → "it learned this itself")
- [ ] Buffer for breakage; record a backup demo video
- [ ] **Milestone:** demo runs end-to-end, twice, without intervention

---

## 10. Risks & mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| PPO doesn't converge in time | Medium | Tiny obs space + proven SB3 defaults; baseline-beating bar is low |
| Reward shaping eats all the time | Medium | Timebox to Day 2–3; ship whatever beats baseline |
| Frontend overruns | High | Pygame fallback if web stalls; record backup video Day 4 |
| Scope creep (multi-intersection) | High | It's a stretch goal only; honor the Day-2 freeze |
| Demo breaks live | Medium | Pre-recorded backup video; run from saved checkpoints, never train live |

---

## 11. Demo script (2 minutes)

1. **Hook (15s):** "Fixed traffic lights waste billions of hours. Watch what happens when an AI controls them instead."
2. **Before (20s):** Run fixed-timer baseline — queues pile up. Point at the wait-time counter climbing.
3. **After (40s):** Toggle to the trained agent — flow smooths out. Counter drops. Show the % improvement.
4. **The reveal (30s):** "Nobody told it the rules. It learned this from scratch in N minutes of training." Show the reward curve climbing.
5. **Close (15s):** Stretch vision — multi-intersection city-scale coordination.

---

## 12. Swapping the concept

If the team prefers a different theme, the entire framework above holds — only Section 5 changes. Swap-in candidates with the same "visual + relatable + survives a mediocre agent" properties:
- **Warehouse robot** picking/routing items
- **Energy grid balancer** (supply vs. demand)
- **Lunar-lander-style** custom landing game
- **Resource/colony manager** mini-game

Keep the observation space small, the action space discrete, and the before/after visual — those are the properties that make it shippable in 4 days.
