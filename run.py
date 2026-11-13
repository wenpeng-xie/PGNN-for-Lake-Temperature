#!/usr/bin/env python3
"""Dispatch a documented analysis, training or final-figure workflow.
Usage: python run.py list | python run.py <task> --help
"""
from __future__ import annotations
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    tasks=json.loads((ROOT/'configs/entrypoints.json').read_text())
    if len(sys.argv)<2 or sys.argv[1] in ('list','--help','-h'):
        print(__doc__+'\n')
        for key,path in tasks.items():print(f'{key:30s} {path}')
        return 0
    name=sys.argv[1]
    if name not in tasks:raise SystemExit(f'Unknown task {name!r}. Run python run.py list.')
    env=os.environ.copy();env.setdefault('MPLBACKEND','Agg')
    command=[sys.executable,str(ROOT/tasks[name]),*sys.argv[2:]]
    return subprocess.call(command,env=env)

if __name__=='__main__':raise SystemExit(main())
