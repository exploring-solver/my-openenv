"""
Baseline inference script for SupportEnv.

Runs an LLM agent against all 3 tasks (5 tickets each) and emits the
mandatory [START]/[STEP]/[END] stdout format.

Environment variables:
  API_BASE_URL   LLM endpoint            (default: https://router.huggingface.co/v1)
  MODEL_NAME     Model identifier         (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       API key
    OPENENV_BASE_URL  SupportEnv server URL (preferred)
    API_BASE_URL_ENV  SupportEnv server URL (backward compatible alias)
"""
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
ENV_BASE_URL = (
    os.getenv("OPENENV_BASE_URL")
    or os.getenv("API_BASE_URL_ENV")
    or "http://localhost:7860"
)

TEMPERATURE = 0.3
MAX_TOKENS = 1024
BENCHMARK = "supportenv"

TASKS = [
    {"task_id": "task1", "name": "Ticket Classification", "tickets": 5},
    {"task_id": "task2", "name": "Information Extraction", "tickets": 5},
    {"task_id": "task3", "name": "Resolution Generation", "tickets": 5},
]


# ---------------------------------------------------------------------------
# Logging helpers (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_request(method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
    url = f"{ENV_BASE_URL}{endpoint}"
    resp = requests.request(method, url, timeout=30, **kwargs)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM prompts per task
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "task1": (
        "You are an expert customer support triage agent.\n"
        "Given a support ticket, classify it by:\n"
        "  category: one of billing | technical | account | feature_request | complaint | general\n"
        "  priority: one of low | medium | high | critical\n\n"
        "Respond with ONLY valid JSON:\n"
        '{"action_type": "classify", "category": "<category>", "priority": "<priority>"}'
    ),
    "task2": (
        "You are an expert information extraction agent for customer support.\n"
        "Given a support ticket, extract ALL structured entities and identify required actions.\n\n"
        "Respond with ONLY valid JSON:\n"
        '{"action_type": "extract", "extracted_entities": {"key": "value", ...}, '
        '"required_actions": ["action1", "action2", ...]}'
    ),
    "task3": (
        "You are an expert customer support resolution agent.\n"
        "Given a support ticket, write a professional customer-facing response and "
        "list the internal resolution steps.\n\n"
        "Requirements:\n"
        "- response_text: Professional, empathetic response (80+ chars)\n"
        "- resolution_steps: Ordered list of internal action identifiers\n"
        "- If the ticket is urgent, acknowledge urgency and provide a timeline\n"
        "- If appropriate, include an apology\n\n"
        "Respond with ONLY valid JSON:\n"
        '{"action_type": "respond", "response_text": "...", '
        '"resolution_steps": ["step1", "step2", ...]}'
    ),
}


def build_user_prompt(task_id: str, ticket: Dict[str, Any]) -> str:
    parts = [
        f"Ticket ID: {ticket['ticket_id']}",
        f"Subject: {ticket['subject']}",
        f"Body: {ticket['body']}",
        f"Customer Tier: {ticket['customer_tier']}",
        f"Account Age: {ticket['account_age_days']} days",
        f"Previous Tickets: {ticket['previous_tickets']}",
    ]
    if ticket.get("attachments"):
        parts.append(f"Attachments: {', '.join(ticket['attachments'])}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, task_id: str, ticket: Dict[str, Any]) -> Dict[str, Any]:
    """Call the LLM and parse its JSON response into an action dict."""
    system_prompt = SYSTEM_PROMPTS[task_id]
    user_prompt = build_user_prompt(task_id, ticket)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        return _parse_json(text, task_id)
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", file=sys.stderr, flush=True)
        return _fallback_action(task_id)


def _parse_json(text: str, task_id: str) -> Dict[str, Any]:
    """Extract JSON from model output, handling markdown fences."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        print(f"[DEBUG] JSON parse failed: {text[:120]}", file=sys.stderr, flush=True)
        return _fallback_action(task_id)


def _fallback_action(task_id: str) -> Dict[str, Any]:
    """Deterministic fallback when LLM fails."""
    if task_id == "task1":
        return {"action_type": "classify", "category": "general", "priority": "medium"}
    elif task_id == "task2":
        return {"action_type": "extract", "extracted_entities": {}, "required_actions": []}
    return {"action_type": "respond", "response_text": "Thank you for contacting support. We are looking into this.", "resolution_steps": []}


# ---------------------------------------------------------------------------
# Run one episode
# ---------------------------------------------------------------------------

def run_episode(
    client: OpenAI, task_id: str, task_name: str, ticket_index: int
) -> Dict[str, Any]:
    """Run a single episode: reset → action → submit → grade."""
    log_start(task=f"{task_name}-ticket{ticket_index}", env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    error_msg: Optional[str] = None

    try:
        # Reset
        obs = env_request("POST", "/reset", json={
            "task_id": task_id, "ticket_index": ticket_index
        })
        episode_id = obs["episode_id"]
        ticket = obs["ticket"]

        # Step 1: LLM generates the action
        action_data = call_llm(client, task_id, ticket)
        result = env_request("POST", "/step", json={
            "episode_id": episode_id, "action": action_data
        })
        steps_taken = 1
        reward_val = result["reward"]["step_reward"]
        rewards.append(reward_val)
        done = result["done"]
        action_summary = _action_summary(action_data)
        log_step(step=1, action=action_summary, reward=reward_val, done=done, error=error_msg)

        # Step 2: Submit if not already done
        if not done:
            submit_result = env_request("POST", "/step", json={
                "episode_id": episode_id,
                "action": {"action_type": "submit"},
            })
            steps_taken = 2
            reward_val = submit_result["reward"]["step_reward"]
            rewards.append(reward_val)
            done = submit_result["done"]
            log_step(step=2, action="submit()", reward=reward_val, done=done, error=None)

        # Grade
        grade = env_request("POST", "/grader", json={"episode_id": episode_id})
        final_score = grade["score"]
        success = final_score >= 0.5

    except Exception as exc:
        error_msg = str(exc)
        print(f"[DEBUG] Episode error: {exc}", file=sys.stderr, flush=True)

    log_end(success=success, steps=steps_taken, rewards=rewards)

    return {
        "task_id": task_id,
        "ticket_index": ticket_index,
        "steps": steps_taken,
        "rewards": rewards,
        "success": success,
    }


def _action_summary(action: Dict[str, Any]) -> str:
    atype = action.get("action_type", "unknown")
    if atype == "classify":
        return f"classify({action.get('category')},{action.get('priority')})"
    elif atype == "extract":
        ents = action.get("extracted_entities") or {}
        acts = action.get("required_actions") or []
        return f"extract({len(ents)}ents,{len(acts)}acts)"
    elif atype == "respond":
        tlen = len(action.get("response_text") or "")
        slen = len(action.get("resolution_steps") or [])
        return f"respond({tlen}chars,{slen}steps)"
    return atype


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    results = []
    for task_info in TASKS:
        task_id = task_info["task_id"]
        task_name = task_info["name"]
        num_tickets = task_info["tickets"]

        for ticket_idx in range(num_tickets):
            result = run_episode(client, task_id, task_name, ticket_idx)
            results.append(result)
            time.sleep(0.5)  # rate-limit courtesy

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("BASELINE RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        total_r = sum(r["rewards"])
        print(
            f"  {r['task_id']} ticket={r['ticket_index']}  "
            f"steps={r['steps']}  reward={total_r:.2f}  {status}",
            flush=True,
        )


if __name__ == "__main__":
    main()
