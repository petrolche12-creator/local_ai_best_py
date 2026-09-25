"""Запуск из папки проекта: python main.py"""

from __future__ import annotations

import sys


def _bootstrap() -> None:
    if sys.version_info < (3, 10):
        print("Нужен Python 3.10 или новее.")
        raise SystemExit(1)
    try:
        from localai.app import main
    except ImportError as exc:
        print("Не хватает зависимостей.")
        print("  python -m venv .venv")
        print("  source .venv/bin/activate")
        print("  pip install -r requirements.txt")
        print(f"({exc})")
        raise SystemExit(1) from exc
    main()


if __name__ == "__main__":
    _bootstrap()
