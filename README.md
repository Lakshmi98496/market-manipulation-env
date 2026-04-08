# Market Manipulation Detection — OpenEnv

> **OpenEnv Hackathon submission** | Finance / Compliance domain  
> Real-world RL environment for detecting market manipulation in live order books.

---

## Overview

An AI agent monitors a **simulated live order book** and must detect suspicious
trading patterns used by bad actors to manipulate asset prices.  
This mirrors actual surveillance systems deployed by exchanges and regulators
like the **SEC** and **SEBI**.

The agent sees a rolling L2 order book snapshot (5 price levels × 10 ticks),
a trade tape, and derived signals (imbalance, cancel rate, spread).  
At each step it decides: **ignore**, **soft_flag**, or **escalate**.

---

## Manipulation Patterns

| Pattern | Description | Signal |
|---|---|---|
| **Spoofing** | Large phantom orders placed then cancelled to push price | High cancel rate + sudden imbalance |
| **Layering** | Multiple stacked orders creating an artificial wall | Uniform large sizes at successive levels |
| **Wash trading** | Self-matched buy/sell pairs to fake volume | Repeated identical price/size trades in tape |
| **None** | Legitimate HFT / market-maker activity | Normal signals |

---

## Action Space

```python
ManipulationAction(
    decision:     "ignore" | "soft_flag" | "escalate",
    pattern_type: "spoofing" | "layering" | "wash_trading" | "none",
    confidence:   float  # 0.0 – 1.0
)
```

## Observation Space

```python
ManipulationObservation(
    bid_levels:      List[PriceLevel]   # 5 levels × 10 ticks
    ask_levels:      List[PriceLevel]   # 5 levels × 10 ticks
    trade_tape:      List[Trade]        # last 20 trades
    order_imbalance: float              # [-1, 1]
    cancel_rate:     float              # [0, 1]
    spread:          float              # dollars
    mid_price:       float              # dollars
    step_number:     int
    task_name:       str
    context_hint:    str                # plain English summary
)
```

---

## Tasks

### Easy — `spoofing_detection`
- Single manipulator doing textbook spoofing
- Clean signal, low noise, 15 steps
- Success threshold: score ≥ 0.3

### Medium — `layering_wash_detection`
- Two simultaneous patterns (layering + wash trading)
- Legitimate HFT noise mixed in, 20 steps
- Agent must triage and classify

### Hard — `adaptive_adversary_detection`
- Adaptive adversary shifts strategy based on agent's flag history
- Market regime switches (calm → volatile) at step 12
- High false-positive risk from market makers, 25 steps

---

## Reward Function

```
reward = partial_credit(decision, true_pattern)   # 0.00–0.60
       + pattern_bonus (if correct pattern named) # +0.20
       + escalation_bonus (for clear crimes)      # +0.10
       + confidence_calibration                   # 0.00–0.10
       − false_positive_penalty                   # −0.30
       (clamped to [0.0, 1.0])
```

Key design:
- **Partial progress signal at every step** — never purely 0/1
- **False positive penalty** — flagging legitimate market makers costs score
- **Confidence calibration** — over-confident wrong answers are penalised

---

## Setup

### Local development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server.env:app --host 0.0.0.0 --port 7860 --reload

# Run graders (verify scores in [0, 1])
python -m tasks.graders

# Run inference (requires LLM endpoint)
export HF_TOKEN=your_token
export HF_SPACE_URL=http://localhost:7860
python inference.py
```

### Docker

```bash
docker build -t market-manipulation-env .
docker run -p 7860:7860 market-manipulation-env
```

### Pre-submission validation

```bash
./validate-submission.sh https://your-space.hf.space .
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/reset` | Start new episode. Body: `{"task": "spoofing_detection", "seed": 42}` |
| POST | `/step`  | Take action. Body: `{"action": {"decision": "escalate", ...}}` |
| GET  | `/state` | Current episode state |
| GET  | `/tasks` | List all tasks |
| GET  | `/health`| Health check |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | Yes | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | Yes | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | Yes | — | HuggingFace / API key |
| `HF_SPACE_URL` | No | `http://localhost:7860` | Environment server URL |
| `TASK_NAME` | No | `spoofing_detection` | Task to run |

---

## Project Structure

```
market-manipulation-env/
├── inference.py          # Main inference script (hackathon entry point)
├── openenv.yaml          # OpenEnv spec
├── Dockerfile            # HF Spaces deployment
├── requirements.txt
├── README.md
├── server/
│   ├── env.py            # FastAPI app + OpenEnv endpoints
│   ├── models.py         # Pydantic action/observation models
│   ├── simulator.py      # Order book simulator + pattern injector
│   └── reward.py         # Reward function with partial credit
└── tasks/
    └── graders.py        # Task graders (easy/medium/hard)
```

---

## Real-World Relevance

This environment models the exact decision problem faced by:
- **Exchange surveillance teams** (NASDAQ, NSE, BSE)
- **Regulator alert systems** (SEC MIDAS, SEBI surveillance)
- **Compliance AI** at hedge funds and brokerages

The three manipulation patterns (spoofing, layering, wash trading) are
all **illegal** under the Securities Exchange Act and SEBI regulations,
and detecting them in real-time is an active area of fintech AI research.
