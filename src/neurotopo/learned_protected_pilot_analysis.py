import argparse
from pathlib import Path
from .learned_protected_pilot import analyze
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);analyze(p.parse_args().run)
