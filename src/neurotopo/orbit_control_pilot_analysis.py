"""Read-only analysis entry for the staged control pilot."""
import argparse
from pathlib import Path
from .orbit_control_pilot import analyze

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    analyze(parser.parse_args().run)

if __name__=='__main__':main()
