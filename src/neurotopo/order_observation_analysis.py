"""Immutable analysis entrypoint for ORDER-OBS qualification."""
import argparse
from pathlib import Path
from .order_observation import analyze

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    analyze(parser.parse_args().run)
