# gomoku/cli.py
import argparse, importlib, inspect, pkgutil, sys, json
from typing import List, Type, Optional, Tuple

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
            continue
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
    found.sort(key=lambda c: f"{c.__module__}.{c.__name__}")
    return found

def _import_class(path: str):
    mod, _, name = path.rpartition(".")
    if not mod:
        raise ValueError(f"Invalid class path: {path}")
    m = importlib.import_module(mod)
    return getattr(m, name)

def _parse_agent_spec(spec: str) -> Tuple[type, str]:
    # "<fqcn>:<display>"  or  "<fqcn>"
    if ":" in spec:
        fqcn, nick = spec.split(":", 1)
    else:
        fqcn, nick = spec, spec.rsplit(".", 1)[-1]
    cls = _import_class(fqcn)
    return cls, nick

async def _play_once(a: Agent, b: Agent, board_size: int = 8, max_turns: int = 200, log=None):
    # Minimal runner using core APIs
    from .core.game import GomokuGame  # adjust if your path differs
    game = GomokuGame(board_size=board_size)
    state = game.new_game()
    turn = 0
    history = []
    while not state.is_terminal() and turn < max_turns:
        agent = a if state.current_player.value == "X" else b
        move = await agent.get_move(state)
        state = game.apply_move(state, move)
        history.append({"turn": turn + 1, "player": state.prev_player.value, "move": move})
        turn += 1
        # pretty print board & last move
        print(state.format_board())
        print(f"Move: {move} (turn {turn})\n")
    result = {"terminal": state.is_terminal(), "winner": state.get_winner() if hasattr(state, "get_winner") else None}
    if log is not None:
        payload = {"history": history, "result": result}
        with open(log, "w") as f:
            json.dump(payload, f, indent=2)
    return result

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

def _cmd_play(args: argparse.Namespace) -> int:
    # Instantiate the two agents
    cls_a, nick_a = _parse_agent_spec(args.agent_a)
    cls_b, nick_b = _parse_agent_spec(args.agent_b)
    a: Agent = cls_a(nick_a)
    b: Agent = cls_b(nick_b)

    # Run one game (async)
    import asyncio
    asyncio.run(_play_once(a, b, board_size=args.size, max_turns=args.max_turns, log=args.log))
    if args.html:
        print("[warn] --html requested but HTML renderer not wired in this minimal CLI.")
    return 0

def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="gomoku", description="Gomoku CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List available agents")
    p_list.add_argument("--detailed", action="store_true", help="Show detailed metadata")
    p_list.set_defaults(func=_cmd_list)

    p_play = sub.add_parser("play", help="Play a game between two agents")
    p_play.add_argument("agent_a", help="FQCN[:Display] for player X")
    p_play.add_argument("agent_b", help="FQCN[:Display] for player O")
    p_play.add_argument("--size", type=int, default=8, help="Board size (default: 8)")
    p_play.add_argument("--max-turns", type=int, default=200, help="Safety cap on turns")
    p_play.add_argument("--log", type=str, default=None, help="Write JSON log to file")
    p_play.add_argument("--html", action="store_true", help="(placeholder) also emit HTML")
    p_play.set_defaults(func=_cmd_play)

    args = p.parse_args(argv)
    return args.func(args)
