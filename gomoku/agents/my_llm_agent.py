# gomoku/agents/my_llm_agent.py
"""
Stronger **prompt-only** Gomoku agent (8×8, five-in-a-row)
==========================================================

**v5.1 – DeepSeek-R1 reasoning + deterministic block/win**
----------------------------------------------------------

*   Deterministic **tactical layer** first (win-now / must-block), O(N²).
*   Otherwise query **DeepSeek-R1-0528-Qwen3-8B** for strategic move.
*   Accept models that emit <think>…</think> and long CoT; we strip
    all reasoning and extract the first JSON object containing row/col.
*   Greedy decoding for stability; no sampling.

Drop-in replacement; no interface changes.
"""
from __future__ import annotations

import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ────────────────────────────────────────────────────────────
# Keep Transformers quiet if present; do NOT import it yet.
# ────────────────────────────────────────────────────────────
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def _ensure_transformers():
    """
    Lazy/optional import for `transformers` to avoid import-time failures
    when extras aren't installed. Call only when the agent is actually used.
    """
    try:
        import transformers as _t  # type: ignore
        try:
            warnings.filterwarnings(
                "ignore", category=UserWarning, module=r"transformers"
            )
            _t.logging.set_verbosity_error()
        except Exception:
            pass
        return _t
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "MyLLMGomokuAgent requires the 'transformers' extra. "
            "Install the repo with extras: pip install -e .[huggingface]"
        ) from e


# ────────────────────────────────────────────────────────────
# Framework imports (with fallbacks for editors/notebooks)
# ────────────────────────────────────────────────────────────
try:
    from .base import Agent
    from ..core.models import GameState
    from ..llm.huggingface_client import HuggingFaceClient
except Exception:  # pragma: no cover – editor stub

    class Agent:  # type: ignore
        def __init__(self, agent_id: str):
            self.agent_id = agent_id

    class GameState:  # type: ignore
        board_size: int = 8
        board: List[List[str]]
        current_player: object

        def format_board(self, *_):
            return ""

        def get_legal_moves(self):
            return [(0, 0)]

    class HuggingFaceClient:  # type: ignore
        def __init__(self, **_):
            self.generation_kwargs, self.generation_config = {}, type("GC", (), {})()
            self.model = type("M", (), {"config": type("C", (), {})()})()

        async def complete(self, _msgs):
            return '{"row":1,"col":1}'


class MyLLMGomokuAgent(Agent):
    """Prompt-driven Gomoku agent with deterministic tactical layer (v5.1)."""

    # UPDATED: DeepSeek-R1-0528 distilled on Qwen3-8B
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    MAX_ATTEMPTS = 4  # one try + up to 3 retries (R1 can be verbose)

    # ──────────────────────────────────────────────────────
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self._setup()

    # ──────────────────────────────────────────────────────
    def _setup(self):
        # Ensure optional dependency only when the agent is actually instantiated.
        _ = _ensure_transformers()

        # Greedy, short-ish outputs; R1 doesn't need beams for our JSON task.
        hf_kwargs: Dict[str, Any] = {
            "model": self.MODEL_NAME,
            "device": "auto",
            "trust_remote_code": True,        # many Qwen-derived chat models require this
            "max_new_tokens": 96,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.05,
        }
        self.llm_client = HuggingFaceClient(**hf_kwargs)
        self._purge_sampling_keys()

        examples = (
            "# WIN – (2,4) completes five → {\"row\":2,\"col\":4}\n"
            "# WIN(1-based) – (5,1) human input → {\"row\":5,\"col\":1}\n"
            "# BLOCK – swap axis (returns 3,6) → {\"row\":3,\"col\":6}\n"
            "# FALL-BACK – if unsure reply {\"row\":4,\"col\":4}"
        )

        # Strong guardrails for R1-style outputs:
        self.system_prompt = (
            "You are a Gomoku grand-master (8×8, five in a row).\n"
            "Respond with **exactly one JSON object on the last line**: "
            "{\"row\":<1-8>,\"col\":<1-8>}.\n"
            "If you need scratch work, put it inside <think>...</think> and never after the final JSON.\n"
            "Priorities: 1 Win • 2 Block • 3 Fork • 4 Open-four • 5 Centre.\n\n"
            + examples
            + "\n\n"
            "Think silently after ##SCRATCHPAD##. If you output anything else by mistake, "
            "reply {\"retry\":true}."
        )

    # ──────────────────────────────────────────────────────
    async def get_move(self, state: GameState) -> Tuple[int, int]:
        legal_moves: Sequence[Tuple[int, int]] = state.get_legal_moves()
        legal_set = set(legal_moves)
        size = state.board_size
        turn_no = size * size - len(legal_moves)

        # 1️⃣ deterministic win / block check -----------------------------
        must_play = self._tactical_forcing_move(state, legal_set)
        if must_play is not None:
            return must_play

        # 2️⃣ otherwise query the language model --------------------------
        user_msg = (
            f"Turn {turn_no}. You play '{state.current_player.value}'.\n"
            f"Board:\n{state.format_board()}\n"
            f"Empty: {sorted(legal_moves)}\n"
            "##SCRATCHPAD##"
        )

        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

        for _ in range(self.MAX_ATTEMPTS):
            move = await self._query_llm(msgs, legal_set, size)
            if move is not None:
                return move
            # nudge the model to return *only* JSON next round
            msgs.append({"role": "assistant", "content": '{"retry":true}'})

        # Last-ditch deterministic move – always legal
        return legal_moves[0]

    # ──────────────────────────────────────────────────────
    def _tactical_forcing_move(
        self, state: GameState, legal_set: set
    ) -> Optional[Tuple[int, int]]:
        """Return an immediate **win** or **must-block** move if present."""
        me = state.current_player.value
        opp = "O" if me == "X" else "X"
        board = state.board
        size = state.board_size

        def scan(char: str) -> Optional[Tuple[int, int]]:
            # helper scanning 4+1 window; returns the empty cell if exactly
            # four `char` and one empty, else None
            for r in range(size):
                for c in range(size - 4):  # horizontal
                    window = [(r, c + i) for i in range(5)]
                    res = self._window_match(window, board, char)
                    if res and res in legal_set:
                        return res
            for c in range(size):
                for r in range(size - 4):  # vertical
                    window = [(r + i, c) for i in range(5)]
                    res = self._window_match(window, board, char)
                    if res and res in legal_set:
                        return res
            for r in range(size - 4):
                for c in range(size - 4):  # diag ↘︎
                    window = [(r + i, c + i) for i in range(5)]
                    res = self._window_match(window, board, char)
                    if res and res in legal_set:
                        return res
            for r in range(4, size):
                for c in range(size - 4):  # diag ↗︎
                    window = [(r - i, c + i) for i in range(5)]
                    res = self._window_match(window, board, char)
                    if res and res in legal_set:
                        return res
            return None

        # first look for our own winning move
        win_move = scan(me)
        if win_move:
            return win_move
        # then look for blocks against opponent four-in-a-row
        return scan(opp)

    @staticmethod
    def _window_match(
        window: List[Tuple[int, int]], board, char: str
    ) -> Optional[Tuple[int, int]]:
        chars = [board[r][c] for r, c in window]
        if chars.count(char) == 4 and chars.count(".") == 1:
            idx = chars.index(".")
            return window[idx]
        return None

    # ──────────────────────────────────────────────────────
    @staticmethod
    def _strip_reasoning(txt: str) -> str:
        """Remove any <think>...</think> blocks (DeepSeek-R1 style)."""
        return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL | re.IGNORECASE)

    # ──────────────────────────────────────────────────────
    async def _query_llm(
        self, msgs, legal_set: set, size: int
    ) -> Optional[Tuple[int, int]]:
        raw = await self.llm_client.complete(msgs)
        raw = self._strip_reasoning(str(raw))
        move = self._parse_move(raw)
        if move is None:
            return None

        r, c = move
        # ── build *all* plausible normalisations ──
        candidates = set()
        # try independent 0/1-based adjustments + axis swap
        for dr in (r, r - 1):
            for dc in (c, c - 1):
                for rr, cc in ((dr, dc), (dc, dr)):
                    if 0 <= rr < size and 0 <= cc < size:
                        candidates.add((rr, cc))
        # pick the *first* legal candidate (stable order for determinism)
        for cand in sorted(candidates):
            if cand in legal_set:
                return cand
        return None

    # ──────────────────────────────────────────────────────
    @staticmethod
    def _parse_move(raw: str) -> Optional[Tuple[int, int]]:
        """
        Accept many JSON-ish formats and return a (row, col) **0-based** tuple.
        Strategy:
          1) Find the first JSON object anywhere with keys row/col.
          2) Otherwise accept a list like [r, c].
          3) Otherwise a plain "r, c" pair.
        """
        txt = raw.strip()

        # 1) Robust object search anywhere in the text
        obj_pat = re.compile(
            r"\{[^{}]*?(?:\"row\"|'row')\s*:\s*(-?\d+)[^{}]*?(?:\"col\"|'col')\s*:\s*(-?\d+)[^{}]*?\}",
            re.DOTALL | re.IGNORECASE,
        )
        m = obj_pat.search(txt)
        if m:
            r, c = int(m.group(1)), int(m.group(2))
            return (r - 1, c - 1)

        # 2) List-like
        try:
            if "[" in txt and "]" in txt:
                arr = json.loads(txt[txt.find("[") : txt.rfind("]") + 1])
                if isinstance(arr, (list, tuple)) and len(arr) >= 2:
                    return (int(arr[0]) - 1, int(arr[1]) - 1)
        except Exception:
            pass

        # 3) Fallback – "num , num" anywhere
        m = re.search(r"(-?\d+)\s*,\s*(-?\d+)", txt)
        if m:
            row, col = map(lambda x: int(x) - 1, m.groups())
            return (row, col)
        return None

    # ──────────────────────────────────────────────────────
    def _purge_sampling_keys(self):
        bad = ("temperature", "top_p", "top_k", "typical_p")
        if hasattr(self.llm_client, "generation_kwargs"):
            for k in bad:
                self.llm_client.generation_kwargs.pop(k, None)
        if hasattr(self.llm_client, "generation_config"):
            for k in bad:
                try:
                    setattr(self.llm_client.generation_config, k, None)
                except Exception:
                    pass
        if hasattr(self.llm_client, "model") and hasattr(self.llm_client.model, "config"):
            for k in bad:
                try:
                    setattr(self.llm_client.model.config, k, None)
                except Exception:
                    pass
