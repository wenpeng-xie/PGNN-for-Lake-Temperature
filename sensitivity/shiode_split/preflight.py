#!/usr/bin/env python3
from pathlib import Path
import argparse
from engine import atomic_json, write_prepared
from settings import configuration, MODELS, EXPECTED_DATA_ONLY_VALID_RATIO, EXPECTED_ORIGINAL90_T0
from lake_data import prepare


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    d=prepare(a.csv)
    r=write_prepared(d,configuration('lstm'),a.out)
    seed_map={m:[dict(member_id=i,actual_seed=s) for i,s in enumerate(configuration(m)['actual_seeds'])] for m in MODELS}
    atomic_json(a.out/'frozen_seed_manifest.json',seed_map)
    for s in ('train','val','test'):
        print(s.upper(),r[s],flush=True)
    print('DATA VERSION: verified original Shiode 2024 ML training record',flush=True)
    print('ROWS:',r['n_rows'],'FULL PERIOD:',r['train']['start'],'->',r['test']['end'],flush=True)
    print('TRAINED TEMPERATURE COLUMNS:',r['temperature_columns'],flush=True)
    print('TRAINED DEPTHS (m):',r['internal_model_depths_m'],flush=True)
    print('DROP/REMAP: none; original Shiode training uses all 11 channels directly',flush=True)
    print('QC EXCLUDED FLAGS:',r['excluded_flags'],flush=True)
    print('PROVENANCE valid ratio:',r['data_only_valid_ratio'],'expected:',EXPECTED_DATA_ONLY_VALID_RATIO,flush=True)
    print('PROVENANCE original 90% T0:',r['original90_T0_fingerprint'],flush=True)
    print('TIMESTAMP DIAGNOSTIC: non-10min steps=',r['timestamp_non10min_steps'],
          'min_step_min=',r['timestamp_min_step_minutes'],'max_step_min=',r['timestamp_max_step_minutes'],flush=True)
    print('TIME HANDLING: original row order preserved; no time reindexing; no water-temperature interpolation',flush=True)
    print('SPLIT-DEPENDENT STATS: meteo normalization + fallback T0 use new train 56% only',flush=True)
    print('MODEL SETTINGS: original Shiode LSTM/rSSM/final-PGNN settings; only chronological split changed',flush=True)
    print('PREFLIGHT_OK',flush=True)

if __name__=='__main__': main()
