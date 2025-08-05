import argparse
import importlib
import inspect
import pkgutil
import sys
from typing import List, Type

from .agents import Agent  # base class


def _discover_agents() -> List[Type[Agent]]:
    import gomoku.agents as agents_pkg

    found = []
    prefix = agents_pkg.__name__ + "."
    for m in pkgutil.walk_packages(agents_pkg.__path__, prefix):
        try:
            mod = importlib.import_module(m.name)
        except Exception:
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, Agent) and obj is not Agent:
                # only keep concrete classes defined in the module
                if obj.__module__ == mod.__name__:
                    found.append(obj)
    # deterministic ordering
    found.sort(key=lambda c: f"{c.__module__}.{c.__name__}")
    return found


def cmd_list(args: argparse.Namespace) -> int:
    agents = _discover_agents()
    print(f"\nDiscovered Agents ({len(agents)}):")
    print("--------------------------------------------------")
    for cls in agents:
        fq = f"{cls.__module__}.{cls.__name__}"
        if args.detailed:
            display = getattr(cls, "display_name", cls.__name__)
            author = getattr(cls, "author", ["Gomoku Framework"])
            version = getattr(cls, "version", "1.0.0")
            desc = getattr(cls, "__doc__", "") or getattr(cls, "description", "")
            desc = desc.strip().splitlines()[0] if desc else ""
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


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="gomoku", description="Gomoku CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List available agents")
    p_list.add_argument("--detailed", action="store_true", help="Show detailed metadata")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
