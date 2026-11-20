"""BFCL (Berkeley Function Calling Leaderboard, pip bfcl-eval) against an OpenAI-compatible server in FC mode
(the server's `tools` / `tool_calls`), under a model name of our own.

  /mnt/HDD/venvs/bfcl/bin/python scripts/run_bfcl.py generate --model tsumi-FC --test-category non_live
  /mnt/HDD/venvs/bfcl/bin/python scripts/run_bfcl.py evaluate --model tsumi-FC --test-category non_live
Env: TSUMI_URL (default http://127.0.0.1:8010), BFCL_PROJECT_ROOT (default /mnt/HDD/eval/quality/bfcl),
BFCL_MODEL (default tsumi-FC; e.g. gguf-Q4_K_M-FC for a llama-server run, with TSUMI_URL pointing at it).
Runs in its own venv (bfcl-eval pins many packages): see scripts/README.md."""
import os
import sys

os.environ.setdefault("BFCL_PROJECT_ROOT", "/mnt/HDD/eval/quality/bfcl")
os.environ.setdefault("OPENAI_BASE_URL", os.environ.get("TSUMI_URL", "http://127.0.0.1:8010").rstrip("/") + "/v1")
os.environ.setdefault("OPENAI_API_KEY", "none")  # the local server does not check it
os.makedirs(os.environ["BFCL_PROJECT_ROOT"], exist_ok=True)

from bfcl_eval.constants import model_config as mc  # noqa: E402
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler  # noqa: E402

for name in {os.environ.get("BFCL_MODEL", "tsumi-FC"), "tsumi-FC", "gguf-Q4_K_M-FC"}:
    mc.MODEL_CONFIG_MAPPING[name] = mc.ModelConfig(
        model_name=name, display_name=f"{name} (Qwen3.8-27B, FC)", url="https://huggingface.co/Qwen/Qwen3.8-27B",
        org="local", license="Apache-2.0", model_handler=OpenAICompletionsHandler, input_price=None,
        output_price=None, is_fc_model=True, underscore_to_dot=True)

from bfcl_eval.__main__ import cli  # noqa: E402

sys.exit(cli())
