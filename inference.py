"""
Inference Script — Market Manipulation Detection
================================================
MANDATORY ENV VARS:
    API_BASE_URL     The API endpoint for the LLM.
    MODEL_NAME       The model identifier to use for inference.
    HF_TOKEN         Your Hugging Face / API key.
    HF_SPACE_URL     URL of the deployed HF Space (default: localhost:7860)
    TASK_NAME        spoofing_detection | layering_wash_detection |
                     adaptive_adversary_detection  (default: spoofing_detection)

STDOUT FORMAT (strict — matches hackathon spec):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""
from __future__ import annotations

import asyncio
import json
import os
import textwrap
from typing import List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY       = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "hf-placeholder")
API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_SPACE_URL  = os.getenv("HF_SPACE_URL", "http://localhost:7860").rstrip("/")

TASK_NAME  = os.getenv("TASK_NAME", os.getenv("MANIPULATION_TASK", "spoofing_detection"))
BENCHMARK  = "market-manipulation-env"

MAX_STEPS = {
    "spoofing_detection": 15,
    "layering_wash_detection": 20,
    "adaptive_adversary_detection": 25,
}.get(TASK_NAME, 15)

SUCCESS_SCORE_THRESHOLD = 0.3
TEMPERATURE  = 0.2   # lower = more consistent decisions
MAX_TOKENS   = 250

# ---------------------------------------------------------------------------
# Logging (exact hackathon format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI surveillance analyst at a financial exchange, monitoring
    the live order book for market manipulation in real time.

    You will receive a detailed Market Surveillance Brief each step.
    Read it carefully — it contains specific signals, anomalies, and context.

    Your job is to make a regulatory decision:
      - ignore     → No credible evidence. Normal market activity.
      - soft_flag  → Suspicious, but not conclusive. Log for review.
                     Use for wash trading (hard to prove definitively).
      - escalate   → Clear, actionable manipulation. Alert regulators NOW.
                     Use for spoofing and layering (phantom orders).

    CRITICAL RULES:
      1. False positives (flagging clean markets) are heavily penalised.
         In volatile regimes, raise your evidence bar significantly.
      2. Wash trading = soft_flag (not escalate — it needs investigation).
      3. Spoofing/layering = escalate (clear, immediate regulatory concern).
      4. Your confidence score affects your reward — be calibrated, not always high.

    Respond with ONLY valid JSON, no other text, no markdown fences:
    {
      "decision": "ignore" | "soft_flag" | "escalate",
      "pattern_type": "spoofing" | "layering" | "wash_trading" | "none",
      "confidence": 0.0-1.0,
      "reasoning": "one concise sentence citing the key signal"
    }
""").strip()


def build_user_prompt(step: int, narrative: str, last_reward: float,
                      history: List[str]) -> str:
    history_block = "\n".join(history[-3:]) if history else "No prior steps."
    return (
        f"{narrative}\n\n"
        f"Last step reward: {last_reward:.2f}\n"
        f"Recent decisions:\n{history_block}\n\n"
        f"Respond with JSON only."
    )

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_model_action(
    client: OpenAI,
    step: int,
    obs: dict,
    last_reward: float,
    history: List[str],
) -> dict:
    narrative = obs.get("context_hint", str(obs))
    user_prompt = build_user_prompt(step, narrative, last_reward, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        # Validate required fields
        assert parsed.get("decision") in ("ignore", "soft_flag", "escalate")
        return parsed
    except (json.JSONDecodeError, AssertionError, KeyError):
        return {
            "decision": "ignore", "pattern_type": "none",
            "confidence": 0.5, "reasoning": "parse error — defaulting to ignore"
        }
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return {
            "decision": "ignore", "pattern_type": "none",
            "confidence": 0.5, "reasoning": "llm error — defaulting to ignore"
        }

# ---------------------------------------------------------------------------
# HTTP env client — sends X-Session-ID for concurrent isolation
# ---------------------------------------------------------------------------

async def env_reset(session: httpx.AsyncClient, task: str,
                    session_id: Optional[str] = None) -> dict:
    headers = {"X-Session-ID": session_id} if session_id else {}
    resp = await session.post(
        f"{HF_SPACE_URL}/reset",
        json={"task": task},
        headers=headers,
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


async def env_step(session: httpx.AsyncClient, action: dict,
                   session_id: Optional[str] = None) -> dict:
    headers = {"X-Session-ID": session_id} if session_id else {}
    resp = await session.post(
        f"{HF_SPACE_URL}/step",
        json={"action": action},
        headers=headers,
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()

# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

async def run_episode(task_name: str) -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    history:    List[str]   = []
    rewards:    List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False
    session_id: Optional[str] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    async with httpx.AsyncClient() as http:
        try:
            # Reset and capture session_id for concurrent isolation
            reset_result = await env_reset(http, task_name)
            session_id   = reset_result.get("session_id")
            obs          = reset_result.get("observation", {})
            last_reward  = 0.0

            for step in range(1, MAX_STEPS + 1):
                action_dict = get_model_action(
                    client, step, obs, last_reward, history
                )

                env_action = {
                    "decision":     action_dict.get("decision", "ignore"),
                    "pattern_type": action_dict.get("pattern_type", "none"),
                    "confidence":   float(action_dict.get("confidence", 0.5)),
                }
                action_str = (
                    f"decision={env_action['decision']} "
                    f"pattern={env_action['pattern_type']} "
                    f"conf={env_action['confidence']:.2f}"
                )

                last_action_error: Optional[str] = None
                try:
                    step_result = await env_step(http, env_action, session_id)
                except Exception as exc:
                    last_action_error = str(exc)
                    step_result = {
                        "observation": obs,
                        "reward": 0.0,
                        "done": True,
                        "last_action_error": str(exc),
                    }

                reward  = float(step_result.get("reward", 0.0))
                done    = bool(step_result.get("done", False))
                obs     = step_result.get("observation", obs)
                err     = step_result.get("last_action_error") or last_action_error

                rewards.append(reward)
                steps_taken = step
                last_reward = reward

                reasoning = action_dict.get("reasoning", "")
                history.append(
                    f"Step {step}: {env_action['decision']} "
                    f"({env_action['pattern_type']}, "
                    f"conf={env_action['confidence']:.2f}) "
                    f"→ reward {reward:+.2f} | {reasoning}"
                )

                log_step(step=step, action=action_str,
                         reward=reward, done=done, error=err)

                if done:
                    break

            score   = sum(rewards) / len(rewards) if rewards else 0.0
            score   = round(max(0.0, min(1.0, score)), 4)
            success = score >= SUCCESS_SCORE_THRESHOLD

        finally:
            log_end(success=success, steps=steps_taken,
                    score=score, rewards=rewards)


async def main() -> None:
    await run_episode(TASK_NAME)


if __name__ == "__main__":
    asyncio.run(main())
