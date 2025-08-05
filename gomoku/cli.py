import argparse
import importlib
import inspect
import pkgutil
import sys
from typing import List, Type

from .agents import Agent  # base class

def _discover_agents() -> list[Type[Agent]]:
    import gomoku.agents as agents_pkg
    found: list[Type[Agent]] = []
    seen: set[tuple[str, str]] = set()
    prefix = agents_pkg.__name__ + "."
    for m in pkgutil.walk_packages(agents_pkg.__path__, prefix):
        try:
            mod = importlib.import_module(m.name)
        except Exception:
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if not (issubclass(obj, Agent) and obj is not Agent):
                continue
            # only keep classes actually defined in this module (avoid re-exports)
            if obj.__module__ != mod.__name__:
                continue
            key = (obj.__module__, obj.__name__)
            if key in seen:
                continue
            seen.add(key)
            found.append(obj)
    found.sort(key=lambda c: f"{c.__module__}.{c.__name__}")
    return found

def _cmd_list(args: argparse.Namespace) -> int:
    agents = _discover_agents()
    print(f"\nDiscovered Agents ({len(agents)}):")
    print("--------------------------------------------------")
    for cls in agents:
        fq = f"{cls.__module__}.{cls.__name__}"
        if args.detailed:
            display = getattr(cls, "display_name", cls.__name__)
            author = getattr(cls, "author", ["Gomoku Framework"])
            version = getattr(cls, "version", "1.0.0")
            desc = (getattr(cls, "__doc__", "") or getattr(cls, "description", "")).strip().splitlines()[0] if (getattr(cls, "__doc__", "") or getattr(cls, "description", "")) else ""
            print(f"✓ {fq}")
            print(f"    Display Name: {display}")
            print(f"    Author: {author}")
            print(f"    Version: {version}")
            print(f"    Description: {desc}")
            print(f"    Agent Class: {fq}")
            print(f"    Source: builtin ({cls.__module__.rsplit('.',1)[0]})\n")
        else:
            print(f"✓ {fq}")
    return 0

def main(argv: List[str] | Non
