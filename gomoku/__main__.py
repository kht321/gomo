# Allows: python -m gomoku list --detailed
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
