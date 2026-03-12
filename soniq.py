#!/usr/bin/env python3
"""Soniq Lab — Audio classification pipeline with web UI.

Run:  python3 soniq.py
      python3 soniq.py --port 9000
      python3 soniq.py --dry-run

First run auto-creates a virtual environment and installs dependencies.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VENV = ROOT / "venv"
DB_PATH = ROOT / "soniq.db"
GENRE_CACHE = ROOT / "output" / "genre_cache.json"

DEPS = [
    "librosa",
    "mutagen",
    "numpy",
    "scipy",
]


def in_venv():
    return sys.prefix != sys.base_prefix


def ensure_venv():
    """Create venv and install deps if needed, then re-exec inside it."""
    if in_venv():
        return  # Already in venv

    venv_python = VENV / "bin" / "python3"

    if not venv_python.exists():
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
        print("Installing dependencies...")
        subprocess.check_call([
            str(venv_python), "-m", "pip", "install", "--quiet", *DEPS,
        ])
        print("Dependencies installed.\n")
    else:
        print("Using existing virtual environment.")

    # Re-exec inside venv
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Soniq Lab — Audio Classification Pipeline")
    parser.add_argument("--port", type=int, default=8877, help="Web UI port (default: 8877)")
    parser.add_argument("--dry-run", action="store_true", help="Extract but don't write tags to files")
    args = parser.parse_args()

    # Suppress noise
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("=" * 50)
    print("  Soniq Lab — v0.5")
    print("  Audio classification pipeline")
    print("=" * 50)

    # Default music folder — check common locations
    default_folder = ""
    for candidate in [
        ROOT / "music",                          # local symlink
        Path.home() / "music-player" / "music",  # MusiCast
        Path.home() / "Music",                   # standard
    ]:
        if candidate.is_dir():
            default_folder = str(candidate)
            break

    # Start server
    from py.server import start
    start(
        db_path=DB_PATH,
        genre_cache_path=GENRE_CACHE,
        port=args.port,
        dry_run=args.dry_run,
        default_folder=default_folder,
    )


if __name__ == "__main__":
    ensure_venv()
    main()
