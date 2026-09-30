"""EXP0018 immutable analysis entrypoint."""
import argparse
from pathlib import Path
from .frozen_binary_probe import analyze

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    analyze(p.parse_args().run)
