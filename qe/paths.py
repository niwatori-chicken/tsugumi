"""Model / weight-cache locations. Override with QE_MODEL_DIR and QE_CACHE_DIR."""
import os

MODEL = os.environ.get("QE_MODEL_DIR", "/mnt/HDD/models/Qwen3.8-27B")
# converted (bfp4/bfp8 .tensorbin) weights are cached under CACHE/<precision>/
CACHE = os.environ.get("QE_CACHE_DIR", "/mnt/HDD/models/qe_cache/Qwen3.8-27B")
