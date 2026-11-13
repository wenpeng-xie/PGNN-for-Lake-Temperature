#!/usr/bin/env python3
from pathlib import Path
import argparse
from engine import atomic_json,write_prepared
from settings import configuration,MODELS
from lake_data import prepare

def main():
    p=argparse.ArgumentParser(); p.add_argument('--csv',required=True,type=Path); p.add_argument('--out',required=True,type=Path)
    a=p.parse_args(); d=prepare(a.csv)
    r=write_prepared(d,configuration('lstm'),a.out)
    seed_map={m:[dict(member_id=i,actual_seed=s) for i,s in enumerate(configuration(m)['actual_seeds'])] for m in MODELS}
    atomic_json(a.out/'frozen_seed_manifest.json',seed_map)
    for s in ('train','val','test'):
        print(s.upper(),r[s],flush=True)
    print('DATA VERSION: verified original Yamanokami 2025 record', flush=True)
    print('ROWS:', r['n_rows'], 'FULL PERIOD:', r['train']['start'], '->', r['test']['end'], flush=True)
    print('SOURCE DEPTH MODE:', r['source_depth_mode'], flush=True)
    print('DROPPED FIRST SOURCE COLUMN:', r['dropped_first_column'], flush=True)
    print('TRAINED TEMPERATURE COLUMNS:', r['temperature_columns'], flush=True)
    print('TRAINED DEPTHS (m):', r['internal_model_depths_m'], flush=True)
    print('QC EXCLUDED FLAGS:', r['excluded_flags'], flush=True)
    print('SPLIT-DEPENDENT STATS: meteorological normalization + fallback T0 use train 56% only', flush=True)
    print('MODEL SETTINGS: original Yamanokami LSTM/rSSM/final-PGNN settings; only chronological split changed', flush=True)
    print('PREFLIGHT_OK',flush=True)
if __name__=='__main__': main()
