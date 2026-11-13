#!/usr/bin/env python3
from pathlib import Path
import argparse,json
from engine import atomic_json,write_prepared
from settings import configuration,MODELS
from shin_data import prepare

def main():
    p=argparse.ArgumentParser(); p.add_argument('--csv',required=True,type=Path); p.add_argument('--out',required=True,type=Path)
    a=p.parse_args(); d=prepare(a.csv)
    r=write_prepared(d,configuration('lstm'),a.out)
    seed_map={m:[dict(member_id=i,actual_seed=s) for i,s in enumerate(configuration(m)['actual_seeds'])] for m in MODELS}
    atomic_json(a.out/'frozen_seed_manifest.json',seed_map)
    for s in ('train','val','test'):
        print(s.upper(),r[s],flush=True)
    print('NORM AND T0: train 56% only; TRAINED CHANNELS:',r['temperature_columns'],flush=True)
    print('PREFLIGHT_OK',flush=True)
if __name__=='__main__': main()
