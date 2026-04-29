"""Compute SHA-256 hashes for files in a target directory tree, skipping __pycache__."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def walk(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if "__pycache__" in p.parts:
            continue
        if p.is_file():
            out.append(p)
    out.sort()
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: _compute_hashes.py <root> [<root>...]", file=sys.stderr)
        sys.exit(2)
    print("| Path | SHA-256 |")
    print("|---|---|")
    project_root = Path.cwd()
    for arg in sys.argv[1:]:
        root = Path(arg).resolve()
        if not root.exists():
            continue
        for p in walk(root):
            rel = p.relative_to(project_root).as_posix()
            digest = sha256_of(p)
            print(f"| `{rel}` | `{digest}` |")


if __name__ == "__main__":
    main()
