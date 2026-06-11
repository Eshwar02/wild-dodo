# ML Research Project — Planning Document

**Date:** 2026-05-25
**Student:** 3rd year CSE, light AI project experience
**Goal:** Pick an innovative ML/AI research direction (not a SaaS/website) that fills a real gap in the field. Fully free — no money spent.

---

## Part 1 — The Landscape: Six Real Unsolved Problems in ML/AI (May 2026)

Most "AI problems" online are wrappers around GPT/Claude. The actual unsolved problems sit one layer below — in the model's behavior, training, evaluation, or deployment. Below are six areas where the gap is real, the solutions are immature, and a motivated undergrad can plausibly contribute.

### 1. Hallucination Detection at Inference Time
- **Gap:** Models confidently invent facts. No reliable way to flag "this token is probably fabricated" *as the model is generating it*.
- **Why unsolved:** Calibration is fundamentally hard. Logits = "next plausible token confidence," not factual confidence. These are different things.
- **Prior art:** Semantic entropy (Farquhar et al., Nature 2024) — requires multiple samples, too expensive in production.
- **Entry point:** Probe internal activations of an open model (Llama 3, Qwen) to find a "hallucination direction" in residual stream. Single GPU sufficient. Publishable.

### 2. Long-Horizon Agent Reliability
- **Gap:** Agents catastrophically fail past ~10–20 reasoning steps. Errors compound. No principled memory architecture exists.
- **Why unsolved:** No "working memory" abstraction for LLMs. RAG is too coarse; full context is too expensive; fine-tuning forgets.
- **Entry point:** Build a benchmark for *task decomposition stability* — same long task 50 times, measure subgoal variance. Benchmark alone is a contribution.

### 3. Evaluation Is Structurally Broken
- **Gap:** Benchmark contamination is rampant. LLM-as-judge has documented biases (position, verbosity, self-preference). MMLU, HumanEval, etc. saturated and partially leaked.
- **Why unsolved:** Building clean, dynamic, contamination-resistant benchmarks is unglamorous, expensive labor. Incentives reward new models, not new evals.
- **Entry point:** Build a *contamination detector* — given a benchmark and model weights, estimate probability the model saw the test set. Foundation: membership inference attacks (Carlini et al.).

### 4. Tokenizer Pathologies
- **Gap:** BPE tokenizers cause real problems: bad arithmetic (9.11 > 9.9), multilingual unfairness (Burmese needs 10× more tokens), glitch tokens (SolidGoldMagikarp), prompt-injection vectors.
- **Why unsolved:** Tokenizers are "good enough" that no one wants to retrain frontier models without one.
- **Prior art:** Byte Latent Transformer (Meta, 2024) showed tokenizer-free models are viable.
- **Entry point:** Don't retrain a frontier model. Build a diagnostic suite. Or train small (100M–1B) byte-level models on specific failure modes.

### 5. Continual Learning Without Catastrophic Forgetting
- **Gap:** Models can't learn new things over time. Every update is full retraining or PEFT — both fragile.
- **Why unsolved:** Backprop destroys what it doesn't reinforce. EWC, replay buffers, parameter isolation all have known failure modes.
- **Entry point:** A *measurement* paper — quantify forgetting curves under different fine-tuning regimes (LoRA vs full FT vs DPO vs continual pretraining). Surprisingly underexplored.

### 6. On-Device Inference for Capable Models
- **Gap:** Running a useful model on a phone is still mostly demo-quality. KV cache eats memory, attention is O(n²), quantization below 4 bits degrades reasoning.
- **Why unsolved:** Memory bandwidth is the bottleneck, not FLOPs. Architectural innovation needed, not just better compression.
- **Entry point:** Implement and benchmark hybrid architectures (Mamba/MoE/sliding-window attention) on consumer hardware. llama.cpp, MLX, candle make this accessible.

---

## Part 2 — Tractability Ranking (For Undergrad, 3–6 month timeline)

1. **Evaluation/contamination tools (#3)** — Least competition, real demand, gets you cited
2. **Hallucination probes (#1)** — Clear methodology, single GPU, hot area
3. **Tokenizer diagnostics (#4)** — Small models, clean experiments
4. **Agent benchmarks (#2)** ← **FINALIZED CHOICE**
5. **On-device hybrids (#6)** — If you like low-level performance work
6. **Continual learning (#5)** — Hardest; needs a mentor

---

## Part 3 — FINALIZED DIRECTION: Agent Benchmarks

### What "agent" means here
An **LLM agent** = an LLM in a loop that can:
1. **Observe** state (file contents, web page, terminal output)
2. **Reason** about what to do
3. **Act** by calling tools (run code, click buttons, edit files)
4. **Repeat** until done or it gives up

Examples: Claude Code, Devin, Operator, browser agents, SWE-agent.

### The central problem: error compounding
A task like "fix this bug in the repo" might need 30 tool calls. At 95% per-step reliability → 0.95^30 ≈ 21% success. Errors compound multiplicatively. **This is what long-horizon failure actually means.**

### Why benchmarking agents is hard (= your opening)
1. **Non-determinism** — Same task, same agent, different outcomes. Most papers report 1 run; statistically meaningless.
2. **Environment drift** — Web agent benchmarks from 2024 broken in 2026 because websites changed. SWE-bench has contamination issues.
3. **Partial credit is fuzzy** — How do you score 80% of a multi-step task?
4. **Cost confounds quality** — Bigger model takes more steps but succeeds more. Hard to compare fairly.
5. **Specification gaming** — Agents find loopholes (WebArena had bug-exploit completions).

Each of these is a research opportunity. Most papers ignore most of them.

### Current state of agent benchmarks (May 2026)
| Benchmark | Domain | Status |
|---|---|---|
| **SWE-bench / SWE-bench Verified** | Fix GitHub issues | Saturating; Claude 4.7 ~75%+. Contamination concerns. |
| **WebArena / VisualWebArena** | Browse real websites | Still hard. Env stability issues. |
| **OSWorld** | Use a real desktop OS | Very hard. ~50% ceiling. |
| **GAIA** | General assistant tasks | Saturating at top end. |
| **AgentBench** | Multi-domain | Older, partially leaked. |
| **τ-bench (tau-bench)** | Customer service dialogues | Tests consistency across multiple runs — closer to ideal. |
| **HAL / HCAST** | Long-horizon (hours) | Newer, METR's time-horizon work. |

**The gap:** almost none measure *what fails and why*. They measure pass/fail. The diagnostic layer is missing.

---

## Part 4 — Four Concrete Project Shapes

### Project A: Agent Reliability Profiler ⭐ RECOMMENDED
Take an existing benchmark (~50 SWE-bench tasks). Run an open agent **20 times per task** (10 on free tier). Measure:
- Variance in success rate per task
- Where do trajectories diverge? (Step 3? Step 15?)
- Are failures correlated (same wrong path) or random (different wrong paths)?

**Output:** Paper saying "current benchmarks understate variance by Nx; here are the tasks where models are actually unreliable vs accidentally lucky." No one has done this cleanly.

### Project B: Subgoal Decomposition Stability
Same task 30 times. Extract the *plan*. Measure:
- How often does it pick the same first subgoal?
- Does the decomposition tree converge?
- Is plan stability correlated with success?

**Why it matters:** Predictive signal for long-horizon failure.

### Project C: Contamination-Resistant Benchmark
~100 procedurally generated tasks — synthetic repos with planted bugs of type Y. Compare leading agents on it vs SWE-bench — show the gap.

**Harder, more original. Closer to top-tier publication.**

### Project D: Error Compounding Model
Mathematical: fit a model predicting agent success from (per-step accuracy, task length, recovery probability). Validate empirically. Connects to METR's time-horizon work.

**Most ambitious, needs more math. Skip for now.**

### Recommendation: Project A
- Uses existing infrastructure
- Result is novel and useful even with null findings
- Teaches the entire agent stack along the way
- 3–4 month scope, single person
- Natural follow-ups (Project B is a literal extension)

---

## Part 5 — FREE STACK (Zero Money Spent)

| Need | Free option |
|---|---|
| LLM inference (cloud) | **Google AI Studio (Gemini 2.0 Flash / 2.5)** — workhorse, very generous free tier |
| Backup LLM | Groq (Llama 3.3 70B, fast), Cerebras, OpenRouter (`:free` variants), GitHub Models, HF Inference API |
| Local LLM | Ollama / llama.cpp — Qwen2.5-Coder-7B at Q4 typical |
| GPU compute | Google Colab (free T4), Kaggle (P100/T4, 30h/week), Lightning AI Studios (22 GPU hr/month) |
| Storage/code | GitHub, HuggingFace Hub |
| Papers | arXiv, alphaXiv, Semantic Scholar |

**Implication:** 10 trials × 25 tasks = 250 agent runs is fully doable on free APIs with patience + provider rotation when rate-limited.

### Hardware reality check
- 8GB RAM, no GPU → 1B–3B local (Qwen2.5-1.5B, Llama 3.2-3B). Slow.
- 16GB RAM, no GPU → 7B quantized (Qwen2.5-Coder-7B Q4). Usable.
- 16GB + 6GB+ VRAM → 7B comfortable, 13B tight.
- No usable machine → Colab/Kaggle for everything.

For Project A: don't need frontier model. Need *consistent* model run many times. 7B coder model is plenty. Or just use Gemini Flash free tier.

---

## Part 6 — Revised Project A (Fully Free)

**Setup:**
- **Agent:** Build a minimal ReAct-style agent yourself (~200–400 lines Python). Don't use OpenHands/SWE-agent — too big to understand in time.
- **Model:** Gemini 2.0 Flash (Google AI Studio free tier). Backup: Groq Llama 3.3 70B.
- **Tasks:** SWE-bench Lite subset (20–30 tasks) OR MiniWoB++ (light web tasks) OR HumanEvalPack.
- **Runs per task:** 10 (rate limit realistic).
- **Total runs:** ~250–300. At ~30 sec each on Gemini Flash → few hours wall time across days.

**Minimal agent loop:**
```python
while not done and step < max_steps:
    response = llm(system_prompt + history)
    action = parse_action(response)
    observation = execute(action)  # run_code, read_file, etc.
    history.append((response, observation))
```

**Measurements:**
1. Pass rate per task (mean ± std)
2. Step count distribution per task
3. Trajectory divergence — at what step do successful and failed runs diverge?
4. Self-consistency — when it succeeds, does it succeed the same way?

**Output:** Clean writeup + open dataset of all trajectories + small repo. Workshop paper or strong arXiv preprint feasible.

---

## Part 7 — Task Suite Options (Free, Low Contamination Risk)

- **HumanEval / MBPP** — Single-function code gen. Easy. Not really "agent" tasks but good warmup.
- **SWE-bench Lite (subset)** — Pick 20 tasks. Docker locally. Real agent benchmark. ⭐
- **MiniWoB++** — 100+ tiny web tasks in browser. Lightweight. Good for variance studies. ⭐
- **MLE-bench (tiny subset)** — ML engineering. Harder setup.
- **Procedurally generated** (→ Project C) — Best for avoiding contamination.

**First project path:** HumanEval as warmup (1 week) → MiniWoB++ or SWE-bench Lite for real experiment.

---

## Part 8 — Two-Week Start Plan

### Week 1 — Foundations
1. Sign up: Google AI Studio, Groq, OpenRouter, GitHub Models. Get API keys. ($0)
2. Install Ollama, pull `qwen2.5-coder:7b` if machine handles it. Test locally.
3. Read these papers (in order, skim — focus on methodology not results):
   - **SWE-bench** (Jimenez et al., 2023) — canonical agent benchmark
   - **SWE-agent** (Yang et al., 2024) — canonical open agent
   - **τ-bench** (Yao et al., 2024) — *introduces multi-run consistency*; direct predecessor
   - **METR's time-horizon paper** (2025) — scaling intuition
   - **WebArena** (Zhou et al., 2023) — environment design
4. Write 100-line ReAct agent that solves HumanEval problems via Gemini Flash. Get end-to-end.

### Week 2 — Get Hands Dirty
1. Pick task suite (MiniWoB++ or 20 SWE-bench Lite tasks).
2. Get 1 task running → scale to 5 → 20.
3. Add logging: every prompt, response, action, observation → JSON file per run.
4. Run each of 5 tasks 10 times. Look at the data. Understanding clicks here.

After Week 2 → scope the real experiment.

---

## Part 9 — Hard Constraints To Internalize

- **Don't build "yet another agent."** Build *measurement*, not capability.
- **Statistics matter.** Mean ± std across 10+ runs → already ahead of 80% of papers.
- **Pick a fixed model.** Compare 5 task types on 1 model, not 5 models on 1 task type. Scope discipline = everything.
- **Reproducibility.** Pin seeds, model versions, environment.
- **Rate limits are real.** Spread runs across days. Rotate providers.
- **Cache aggressively.** Disk is free. Log everything — you'll want trajectories later.

---

## Part 10 — User Profile (Confirmed)

- **Machine:** Dell G15 5530, Fedora Linux
- **RAM:** 16GB (10GB usable)
- **GPU:** RTX 3050 6GB mobile, 90W TDP
- **Python comfort:** 5/10
- **API experience:** Done many projects integrating APIs

**Hardware verdict:**
- RTX 3050 6GB — enough for 7B at Q4 (tight); 3B–4B comfortable with headroom.
- 10GB usable RAM — fine for one local model + browser + editor; avoid running Docker + 7B model simultaneously (will swap).
- Fedora — perfect, no compatibility surprises.

**Practical setup decision:**
- **Primary path:** Gemini 2.0 Flash via free API (faster, no VRAM constraints, doesn't lock up machine).
- **Local fallback:** Qwen2.5-Coder-3B at Q4 via Ollama for offline dev iteration. Skip 7B for now — keep VRAM/RAM headroom.
- API path is faster and free. Local is for understanding + dev iteration only.

---

## Part 11 — Day 1 Plan (2 hours, end-to-end)

### Step 1 — API keys (15 min)
Sign up, no card on file:
1. **Google AI Studio** → https://aistudio.google.com/apikey — primary, Gemini 2.0 Flash, millions of tokens/day free.
2. **Groq** → https://console.groq.com/keys — backup, Llama 3.3 70B, very fast.
3. **OpenRouter** → https://openrouter.ai/keys — rotation, `:free` model variants.

Add to `~/.bashrc`:
```bash
export GEMINI_API_KEY="..."
export GROQ_API_KEY="..."
export OPENROUTER_API_KEY="..."
```
Then `source ~/.bashrc`.

### Step 2 — Project scaffold (10 min)
```bash
cd "/run/media/eshhh/Disk E:/esh_proj_files/claude talk"
mkdir agent-reliability && cd agent-reliability
python -m venv .venv
source .venv/bin/activate
pip install google-genai groq openai python-dotenv
git init
```

Notes:
- `openai` package used because OpenRouter is OpenAI-compatible.
- `google-genai` is the new official Gemini SDK — replaces old `google-generativeai`.

### Step 3 — Smoke test all three providers (15 min)
Create `test_providers.py`:
```python
import os
from google import genai
from groq import Groq
from openai import OpenAI

prompt = "Reply with exactly: pong"

# Gemini
g = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
r = g.models.generate_content(model="gemini-2.0-flash", contents=prompt)
print("Gemini:", r.text.strip())

# Groq
gr = Groq(api_key=os.environ["GROQ_API_KEY"])
r = gr.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
)
print("Groq:", r.choices[0].message.content.strip())

# OpenRouter
o = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
r = o.chat.completions.create(
    model="deepseek/deepseek-chat-v3.1:free",
    messages=[{"role": "user", "content": prompt}],
)
print("OpenRouter:", r.choices[0].message.content.strip())
```
All three should print "pong". Fix any failure before moving on.

### Step 4 — Install Ollama + small local model (20 min)
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2.5-coder:3b
ollama run qwen2.5-coder:3b "Write a Python function that returns the fibonacci sequence up to n"
```
3B at Q4 fits 6GB VRAM comfortably. Skip 7B for now.

### Step 5 — Minimal ReAct agent (45 min)
Create `agent.py`:
```python
"""Minimal ReAct agent. Solves coding tasks via tool calls."""
import json, subprocess, os, re, tempfile
from google import genai

MODEL = "gemini-2.0-flash"
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM = """You are a coding agent. You solve tasks by calling tools.
Available tools:
- write_file(path, content): write content to a file
- run_python(code): execute python code, returns stdout/stderr
- finish(answer): submit your final answer

Respond ONLY with a JSON object of the form:
{"thought": "...", "tool": "tool_name", "args": {...}}
"""

def run_python(code: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    try:
        r = subprocess.run(["python", path], capture_output=True, text=True, timeout=10)
        return (r.stdout + r.stderr)[:2000]
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    finally:
        os.unlink(path)

def write_file(path: str, content: str) -> str:
    with open(path, "w") as f: f.write(content)
    return f"wrote {len(content)} bytes to {path}"

TOOLS = {"run_python": run_python, "write_file": write_file}

def parse_action(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except: return None

def run_agent(task: str, max_steps: int = 15) -> dict:
    history = [f"TASK: {task}"]
    trajectory = []
    for step in range(max_steps):
        prompt = SYSTEM + "\n\n" + "\n\n".join(history)
        resp = client.models.generate_content(model=MODEL, contents=prompt)
        text = resp.text
        action = parse_action(text)
        trajectory.append({"step": step, "response": text, "action": action})
        if not action:
            history.append(f"OBSERVATION: invalid JSON, try again")
            continue
        if action["tool"] == "finish":
            return {"success": True, "answer": action["args"].get("answer"), "steps": step+1, "trajectory": trajectory}
        tool = TOOLS.get(action["tool"])
        if not tool:
            obs = f"unknown tool {action['tool']}"
        else:
            obs = tool(**action["args"])
        history.append(f"THOUGHT: {action.get('thought','')}\nACTION: {action['tool']}({action['args']})\nOBSERVATION: {obs}")
    return {"success": False, "answer": None, "steps": max_steps, "trajectory": trajectory}

if __name__ == "__main__":
    task = "Write a python function `is_prime(n)` that returns True if n is prime, False otherwise. Test it on n=17 and n=15, then finish with the test results as the answer."
    result = run_agent(task)
    print(json.dumps(result, indent=2)[:3000])
```

Run: `python agent.py`. Watch the trajectory — this is what you'll be measuring 1000× later.

### Step 6 — Background reading (15 min while agent runs)
Open **SWE-bench paper** (arXiv 2310.06770). Skim sections 1, 3 (dataset), 5 (evaluation methodology). Get the shape, not the depth.

---

## Part 12 — Day 1 Definition of Done

- ✅ Three API providers working (Gemini, Groq, OpenRouter)
- ✅ Ollama + Qwen-3B running locally
- ✅ Minimal ReAct agent solves a toy coding task end-to-end
- ✅ One full agent trajectory inspected in JSON
- ✅ Skimmed SWE-bench paper

**Day 2 preview:** Add proper logging → run agent on 5 HumanEval problems × 10 runs each → see variance for the first time. That's when the project becomes real.

---

## Summary

**Finalized:** Agent benchmarks, Project A (Agent Reliability Profiler), fully free stack (Gemini Flash primary + Groq/OpenRouter backups + Ollama Qwen-3B local).

**Status:** Day 1 plan ready. User has hardware + skills confirmed. Next: execute Day 1.
