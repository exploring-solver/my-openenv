---
title: SupportEnv
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - customer-support
  - nlp
  - ticket-triage
  - agent-evaluation
pinned: false
---

# SupportEnv

SupportEnv is an OpenEnv-compliant environment for evaluating LLM agents on customer support ticket triage. Each episode presents a realistic support ticket and asks the agent to classify, extract, or resolve it — scored deterministically against ground-truth labels.

## Tasks

| Task | Difficulty | Action | Max Steps |
|------|-----------|--------|-----------|
| Task 1 — Ticket Classification | Easy | `classify` | 3 |
| Task 2 — Information Extraction | Medium | `extract` | 5 |
| Task 3 — Resolution Generation | Hard | `respond` | 8 |

**Task 1 — Ticket Classification (Easy)**  
Assign a `category` (billing / technical / account / feature_request / complaint / general) and `priority` (low / medium / high / critical) to each ticket.

**Task 2 — Information Extraction (Medium)**  
Extract structured entities (IDs, names, amounts, dates) and identify the list of required resolution actions.

**Task 3 — Resolution Generation (Hard)**  
Write a professional customer-facing response and an ordered list of internal resolution steps. Graded on keyword coverage, step completeness, tone adherence, and minimum length.

## Observation Space

Each observation includes:

- `task_id`, `task_description`, `episode_id`
- `ticket` object with `ticket_id`, `subject`, `body`, `customer_tier`, `account_age_days`, `previous_tickets`, `attachments`
- `thread_history` as ordered action summaries
- `available_actions` for the current task state
- `step_number`, `max_steps`
- `hint` (optional guidance)

## Action Space

Supported `action.action_type` values:

- `classify`: requires `category` and `priority`
- `extract`: requires `extracted_entities` and `required_actions`
- `respond`: requires `response_text` and `resolution_steps`
- `submit`: closes the episode and triggers terminal grading

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit an action |
| `GET` | `/state` | Get current episode state |
| `POST` | `/grader` | Grade a finished episode |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/health` | Liveness check |
| `GET` | `/docs` | OpenAPI docs |

### Reset
```json
POST /reset
{"task_id": "task1", "ticket_index": 0}
```

### Step — Task 1 (classify)
```json
POST /step
{
  "episode_id": "<id>",
  "action": {"action_type": "classify", "category": "billing", "priority": "high"}
}
```

### Step — Task 2 (extract)
```json
POST /step
{
  "episode_id": "<id>",
  "action": {
    "action_type": "extract",
    "extracted_entities": {"customer_name": "Alice", "invoice_number": "INV-001"},
    "required_actions": ["issue_refund", "send_corrected_invoice"]
  }
}
```

### Step — Task 3 (respond)
```json
POST /step
{
  "episode_id": "<id>",
  "action": {
    "action_type": "respond",
    "response_text": "Dear customer, we sincerely apologize...",
    "resolution_steps": ["verify_account", "issue_refund", "send_confirmation"]
  }
}
```

### Submit
```json
POST /step
{"episode_id": "<id>", "action": {"action_type": "submit"}}
```

## Scoring

**Task 1:** category match (0.50) + priority match (0.40) + efficiency (0.10)

**Task 2:** entity coverage (0.60) + action coverage (0.30) + no hallucination (0.10)

**Task 3:** keyword coverage (0.30) + step coverage (0.30) + tone compliance (0.25) + length adequate (0.10) + non-empty steps (0.05)

## Running Locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Running the Baseline Agent

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export HF_TOKEN=your_token_here
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

Required environment variables for baseline LLM calls:

- `API_BASE_URL` (default provided in code)
- `MODEL_NAME` (default provided in code)
- `HF_TOKEN` (must be provided)

Environment endpoint variables for the baseline:

- `OPENENV_BASE_URL` (preferred, default `http://localhost:7860`)
- `API_BASE_URL_ENV` (backward-compatible alias)

The baseline emits strict structured stdout lines only:

- `[START] task=<...> env=<...> model=<...>`
- `[STEP] step=<...> action=<...> reward=<...> done=<...> error=<...>`
- `[END] success=<...> steps=<...> rewards=<...>`

## Docker

```bash
docker build -t supportenv .
docker run -p 7860:7860 supportenv
```
