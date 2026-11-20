"""Settings from environment variables: TSUMI_<NAME>, with the old QE_<NAME> still accepted."""
import os


def env(name, default):
    return os.environ.get("TSUMI_" + name, os.environ.get("QE_" + name, default))
