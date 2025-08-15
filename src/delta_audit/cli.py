import argparse
from .runners import run_benchmark, run_quickstart
from .plotting import make_overview

def main():
    p=argparse.ArgumentParser('delta-audit')
    sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('quickstart')
    r=sub.add_parser('run'); r.add_argument('--config',required=True)
    f=sub.add_parser('figures'); f.add_argument('--summary',required=True); f.add_argument('--out',required=True)
    a=p.parse_args()
    if a.cmd=='quickstart': run_quickstart()
    elif a.cmd=='run': run_benchmark(a.config)
    elif a.cmd=='figures': make_overview(a.summary,a.out)
