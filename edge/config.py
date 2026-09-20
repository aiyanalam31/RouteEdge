"""
config.py

Tiny YAML config loader for the edge side. Kept dependency-light (just
PyYAML) and simple on purpose — this is a hackathon project, not a
production config system.
"""

import os

import yaml

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load(path=_DEFAULT_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)
