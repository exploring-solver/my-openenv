# setup.md - SupportEnv Validator-Focused Runbook

## 1. What judges/validator execute

Most checks align to this flow:

1. `POST /reset` on the deployed Space
2. `docker build` from repo root
3. `openenv validate`
4. endpoint contract checks for `/health`, `/reset`, `/step`, `/state`, `/grader`
5. `python inference.py` and stdout format check for `[START]`, `[STEP]`, `[END]`


## 2. File-by-file usage (root)

- `app.py`: FastAPI API surface (`/reset`, `/step`, `/state`, `/tasks`, `/grader`, `/health`)
- `environment.py`: episode lifecycle and reward accumulation (`reset`, `step`, `get_state`, `grade`)
- `graders.py`: deterministic terminal scoring per task with score clamped to `[0.0, 1.0]`
- `data.py`: task metadata and ticket datasets with ground truth labels/entities/steps
- `models.py`: typed Pydantic models used by API and internal state
- `inference.py`: baseline runner; calls the API, logs strict `[START]/[STEP]/[END]`
- `openenv.yaml`: OpenEnv metadata and interface declaration used by validator
- `Dockerfile`: image build/runtime contract for HF Docker Spaces (serves on `7860`)
- `requirements.txt`: runtime dependencies
- `pyproject.toml`: packaging metadata + script entrypoint expected by validator tooling
- `uv.lock`: lockfile required by OpenEnv multi-mode validation path
- `server/app.py`: validator-friendly script entrypoint (`server = server.app:main`)


## 3. Local setup

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## 4. Validation checklist (exact order)

1. OpenEnv validator

```bash
.venv/Scripts/openenv.exe validate
```

2. Docker build

```bash
docker build -t supportenv .
```

3. Run server locally

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

4. API checks

```bash
curl http://127.0.0.1:7860/health
curl -X POST http://127.0.0.1:7860/reset -H "Content-Type: application/json" -d '{"task_id":"task1","ticket_index":0}'
curl -X POST http://127.0.0.1:7860/step -H "Content-Type: application/json" -d '{"episode_id":"<id>","action":{"action_type":"classify","category":"billing","priority":"high"}}'
curl -X POST http://127.0.0.1:7860/state?episode_id=<id>
curl -X POST http://127.0.0.1:7860/grader -H "Content-Type: application/json" -d '{"episode_id":"<id>"}'
```

5. Baseline inference

```bash
python inference.py
```


## 5. Docker and Spaces runtime model

- Build stage installs from `requirements.txt`.
- Runtime command runs Uvicorn: `app:app` on `0.0.0.0:7860`.
- HF Space should set `sdk: docker` and `app_port: 7860` in `README.md` frontmatter.
- Healthcheck points at `/health` to indicate container liveness.
- If Docker daemon is not running locally, `docker build`/`docker run` will fail even if repo is correct.


## 6. Inference variables

- Required for LLM call path:
  - `API_BASE_URL`
  - `MODEL_NAME`
  - `HF_TOKEN`
- Environment endpoint:
  - `OPENENV_BASE_URL` (preferred)
  - `API_BASE_URL_ENV` (legacy alias)


## 7. Example scorer sanity checks

- Task 1: submit `classify` then `submit`, verify non-binary reward and final score in `[0, 1]`
- Task 2: include deterministic entity/action coverage keys from ticket text
- Task 3: include professional response plus ordered resolution steps


## 8. Common failure causes

- Missing `pyproject.toml` or `uv.lock`
- Missing script entrypoint (`server = server.app:main`)
- App not serving on `0.0.0.0:7860`
- Duplicate HF variable/secret names in Space settings
- Invalid or missing `HF_TOKEN` for real LLM inference