# gomoku/cli.py
import argparse
import asyncio
import importlib
import inspect
import json
import pkgutil
import sys
from typing import List, Optional, Type

from .agents import Agent  # base class


# -------- Agent discovery (de-dupe re-exports) ------------------------------
def _discover_agents() -> List[Type[Agent]]:
    import gomoku.agents as agents_pkg

    found: List[Type[Agent]] = []
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
            if obj.__module__ != mod.__name__:
                continue  # ignore re-exports
            key = (obj.__module__, obj.__name__)
            if key in seen:
                continue
            seen.add(key)
            found.append(obj)

    found.sort(key=lambda c: f"{c.__module__}.{c.__name__}")
    return found


def _import_class(path: str):
    mod, _, name = path.rpartition(".")
    if not mod:
        raise ValueError(f"Invalid class path: {path}")
    m = importlib.import_module(mod)
    return getattr(m, name)


def _parse_agent_spec(spec: str):
    # "<fqcn>:<display>"  or  "<fqcn>"
    if ":" in spec:
        fqcn, nick = spec.split(":", 1)
    else:
        fqcn, nick = spec, spec.rsplit(".", 1)[-1]
    cls = _import_class(fqcn)
    return cls, nick


# -------- Game class resolver (robust across repo layouts) ------------------
def _resolve_game_class():
    """
    Try typical locations; if not found, scan gomoku.* for a class that
    exposes new_game() and apply_move(state, move).
    """
    candidates = [
        ("gomoku.core.game", "GomokuGame"),
        ("gomoku.core.engine", "GomokuGame"),
        ("gomoku.core.game", "Game"),
        ("gomoku.arena.game", "GomokuGame"),
        ("gomoku.arena.engine", "GomokuGame"),
    ]
    for mod, name in candidates:
        try:
            m = importlib.import_module(mod)
            return getattr(m, name)
        except Exception:
            pass

    # Fallback: scan all gomoku submodules
    import gomoku as root
    for m in pkgutil.walk_packages(root.__path__, root.__name__ + "."):
        try:
            mod = importlib.import_module(m.name)
        except Exception:
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if getattr(obj, "__name__", "") in {"GomokuGame", "Game"}:
                if hasattr(obj, "new_game") and hasattr(obj, "apply_move"):
                    return obj

    raise ImportError(
        "Could not locate a GomokuGame class. Please adjust _resolve_game_class() "
        "to point to your game's class (module & name)."
    )


# -------- Commands ----------------------------------------------------------
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
            print(f"    Source: builtin ({cls.__module__.rsplit('.',1)[0]})\n")
        else:
            print(f"✓ {fq}")
    return 0


async def _play_once(a: Agent, b: Agent, board_size: int, max_turns: int, log: Optional[str]):
    GameCls = _resolve_game_class()
    # constructor may or may not accept board_size
    try:
        game = GameCls(board_size=board_size)
    except TypeError:
        game = GameCls()

    # game API is expected to have new_game()/apply_move() and state has format_board()/is_terminal()
    state = game.new_game()
    turn = 0
    history = []

    def _fmt_board(s):
        if hasattr(s, "format_board"):
            try:
                return s.format_board()
            except Exception:
                pass
        return getattr(s, "board", s)

    while (not getattr(state, "is_terminal", lambda: False)()) and turn < max_turns:
        agent = a if state.current_player.value == "X" else b
        move = await agent.get_move(state)
        state = game.apply_move(state, move)
        history.append({"turn": turn + 1, "player": state.prev_player.value if hasattr(state, "prev_player") else None, "move": move})
        print(_fmt_board(state))
        print(f"Move: {move} (turn {turn + 1})\n")
        turn += 1

    result = {
        "terminal": getattr(state, "is_terminal", lambda: False)(),
        "winner": state.get_winner() if hasattr(state, "get_winner") else None,
        "turns": turn,
    }
    if log:
        with open(log, "w") as f:
            json.dump({"history": history, "result": result}, f, indent=2)
    return result


def _cmd_play(args: argparse.Namespace) -> int:
    cls_a, nick_a = _parse_agent_spec(args.agent_a)
    cls_b, nick_b = _parse_agent_spec(args.agent_b)
    a: Agent = cls
