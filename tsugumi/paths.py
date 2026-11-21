"""Model / weight-cache locations. Override with TSUGUMI_MODEL_DIR and TSUGUMI_CACHE_DIR."""
import os

from .env import env

MODEL = env("MODEL_DIR", "/mnt/HDD/models/Qwen3.8-27B")
# converted (bfp4/bfp8 .tensorbin) weights are cached under CACHE/<precision>/
CACHE = env("CACHE_DIR", "/mnt/HDD/models/qe_cache/Qwen3.8-27B")
