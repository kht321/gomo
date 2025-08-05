# gomoku/cli.py
import argparse, importlib, inspect, pkgutil, sys
from typing import List, Type
from .agents import Agent

def _discover_agents() -> List[Type[Agent]]:
    import gomoku.agents as agents_pkg
    seen = set()
    found: List[Type[Agent]] = []
    prefix = agents_pkg.__name__ + "."

    for m in pkgutil.walk_packages(agents_pkg.__path__, prefix):
        try:
            mod = importlib.import_module(m.name)
        except Exception:
            continue

        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if not (issubclass(obj, Agent) and obj is not Agent):
                continue
            # Keep only classes defined in their module (avoid package re-exports)
            if obj.__module__ != mod.__name__:
                continue
            key = (obj.__module__, obj.__name__)
            if key in seen:
                continue
            seen.add(key)
            found.append(obj)

    found.sort(key=lambda c: f"{c.__module__}.{c.__name__}")
    return found
