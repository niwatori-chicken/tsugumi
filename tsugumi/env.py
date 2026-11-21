"""Settings from environment variables: TSUGUMI_<NAME>, with the old names TSUMI_<NAME> and QE_<NAME> still accepted."""
import os


def env(name, default):
    for prefix in ("TSUGUMI_", "TSUMI_", "QE_"):
        v = os.environ.get(prefix + name)
        if v is not None:
            return v
    return default
