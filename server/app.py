"""Validator-friendly server entrypoint for SupportEnv."""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Launch the FastAPI app on the Hugging Face expected host/port."""
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":
    main()
