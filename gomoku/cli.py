# gomoku/cli.py
import argparse
import importlib
import inspect
import pkgutil
import sys
from typing import List, Type, Optional

from .agents import Agent  # base class


def _discover_agents() -> List[Type[Agent]]:
    import gomoku.agents as agents_pkg

    found: List[Type[Agent]] = []
    seen: set[tuple[str, str]] = set()
    prefix = agents_pkg.__name__ + "."

    for m in pkgutil.walk_packages(agents_pkg.__path__, prefix):
        try:
            mod = importlib.import_module(m.name)
        except Exception:
            # Skip modules that fail to import (optional deps, etc.)
            continue

        # Collect only concrete Agent subclasses defined in that module
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if not (issubclass(obj, Agent) and obj is not Agent):
                continue
            if obj.__module__ != mod.__name__:
                continue
            key = (obj.__module__, obj.__name__)
            if key in seen:
                continue
            seen.add(key)
            found.append(obj)

    # Deterministic order
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
            desc = (getattr(cls, "__doc__", "") or getattr(cls, "description", "")).strip()
            first_line = desc.splitlines()[0] if desc else ""
            print(f"✓ {fq}")
            print(f"    Display Name: {display}")
            print(f"    Author: {author}")
            print(f"    Version: {version}")
            print(f"    Description: {first_line}")
            print(f"    Agent Class: {fq}")
            print(f"    Source: builtin ({cls.__module__.rsplit('.', 1)[0]})\n")
        else:
            print(f"✓ {fq}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="gomoku", description="Gomoku CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List available agents")
    p_list.add_argument("--detailed", action="store_true", help="Show detailed metadata")
    p_list.set_defaults(func=_cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)
