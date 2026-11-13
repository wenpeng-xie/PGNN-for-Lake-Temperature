#!/usr/bin/env python3
"""Run the original cross-year models with portable paths and frozen member IDs.

Only filesystem routing and member orchestration are provided here. The model,
loss, preprocessing and optimization implementations live in implementations/.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]

def load_source(path: Path, name: str):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ImportError(str(path))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site',choices=['yamanokami','shiode'],required=True)
    p.add_argument('--model',choices=['lstm','rssm','pgnn'],required=True)
    p.add_argument('--phase',choices=['train','predict'],required=True)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data'/'observations')
    p.add_argument('--out',type=Path,required=True,help='Member directories are written as seed<actual seed>.')
    p.add_argument('--seeds',type=int,nargs='+',help='Override frozen member list for a documented exploratory run.')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--overwrite',action='store_true',help='Allow an existing checkpoint to be overwritten during training.')
    a=p.parse_args()
    members=json.loads((ROOT/'configs'/'final_members.json').read_text())
    seeds=a.seeds if a.seeds is not None else members[a.site][a.model]
    if len(set(seeds))!=len(seeds):p.error('Duplicate seeds are not allowed.')
    data=a.data_dir.resolve();out=a.out.resolve()
    year='2025' if a.site=='yamanokami' else '2024'
    test_year='2024' if a.site=='yamanokami' else '2025'
    needed=[data/f'{a.site}_{year}_aligned_qc_wide.csv']
    if a.phase=='predict':needed.append(data/f'{a.site}_{test_year}_aligned_qc_wide.csv')
    missing=[str(x) for x in needed if not x.is_file()]
    if missing:raise SystemExit('Missing input files:\n'+'\n'.join(missing)+'\nSee docs/DATA_LAYOUT.md.')
    for seed in seeds:
        ckpt=out/f'seed{seed}'/'best_model.pt'
        if a.phase=='train' and ckpt.exists() and not a.overwrite:
            raise SystemExit(f'Refusing to overwrite {ckpt}; choose a new --out or use --overwrite.')
        if a.phase=='predict' and not ckpt.is_file():raise SystemExit(f'Missing checkpoint: {ckpt}')
    os.environ['JOHRS_DATA_DIR']=str(data)
    os.environ['JOHRS_RUNS_DIR']=str(out)
    os.environ['JOHRS_SELECTED_SEEDS']=json.dumps(seeds)
    os.environ.setdefault('MPLBACKEND','Agg')
    import torch
    torch.set_num_threads(a.threads)
    out.mkdir(parents=True,exist_ok=True)
    record=dict(site=a.site,model=a.model,phase=a.phase,actual_seeds=seeds,
                data_dir=str(data),paper_members=seeds==members[a.site][a.model],
                model_source='unchanged cross-year notebook implementation; paths parameterized')
    (out/f'{a.phase}_invocation.json').write_text(json.dumps(record,indent=2))
    source=Path(__file__).parent/'implementations'/f'{a.site}_{a.model}_{a.phase}.py'
    if a.phase=='train':
        module=load_source(source,f'johrs_{a.site}_{a.model}_train')
        if a.model=='pgnn':module.main()
        else:
            for seed in seeds:module.main(seed)
    else:
        for seed in seeds:
            os.environ['JOHRS_MEMBER_DIR']=str(out/f'seed{seed}')
            os.environ['JOHRS_PREDICT_DIR']=str(out/'predictions'/f'seed{seed}')
            module=load_source(source,f'johrs_{a.site}_{a.model}_predict_seed{seed}')
            if a.model=='pgnn':module.main(seed)
            else:module.main()

if __name__=='__main__':main()
