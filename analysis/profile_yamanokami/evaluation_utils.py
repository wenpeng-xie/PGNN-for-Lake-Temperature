#!/usr/bin/env python3
"""Read-only archive, alignment, and metric helpers. No training or member selection."""
from __future__ import annotations
import argparse, csv, io, json, hashlib, re, tarfile, zipfile
from datetime import datetime
from pathlib import Path
import numpy as np


def sha256(p):
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def load_npz(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def as_time(value):
    if '-' in value: return np.datetime64(value, 'ns')
    for fmt in ('%m/%d/%Y %H:%M', '%m/%d/%Y %H:%M:%S'):
        try: return np.datetime64(datetime.strptime(value,fmt),'ns')
        except ValueError: pass
    raise ValueError('Unrecognized timestamp: '+value)


def boolean(value):
    v=value.strip().lower()
    if v in ('true','1'): return True
    if v in ('false','0',''): return False
    raise ValueError('Invalid QC boolean: '+repr(value))


def csv_rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))


def write_csv(path, rows):
    rows=list(rows)
    if not rows: return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def metrics(pred,obs,mask):
    p=np.asarray(pred,dtype=np.float64)[mask]; o=np.asarray(obs,dtype=np.float64)[mask]
    if len(p)==0: raise ValueError('No valid evaluation samples')
    if not (np.isfinite(p).all() and np.isfinite(o).all()): raise ValueError('Nonfinite on fixed evaluation mask')
    e=p-o; sse=float(e@e); do=o-o.mean(); dp=p-p.mean()
    sst=float(do@do); vp=float(dp@dp)
    return dict(N=len(p), RMSE=float(np.sqrt(sse/len(p))),
                R=float(dp@do/np.sqrt(vp*sst)) if sst>0 and vp>0 else float('nan'),
                NSE=float(1-sse/sst) if sst>0 else float('nan'), Bias=float(e.mean()), SSE=sse)


def model_summary(rows):
    result=[]
    for model in ('GLM','LSTM','rSSM','PGNN'):
        sub=[r for r in rows if r['model']==model]
        assert len(sub)==10, (model,len(sub))
        r=dict(model=model,n_members=len(sub),N=sub[0]['N'])
        assert len({q['N'] for q in sub})==1
        for k in ('RMSE','R','NSE','Bias'):
            x=np.array([q[k] for q in sub]); r[k+'_mean']=float(x.mean()); r[k+'_sd']=float(x.std(ddof=1)); r[k+'_median']=float(np.median(x))
        result.append(r)
    return result


def align_time(source,target):
    if np.any(np.diff(source.astype('datetime64[ns]').astype(np.int64))<=0):
        raise ValueError('Time axis duplicated or not strictly increasing')
    idx=np.searchsorted(source,target)
    if (idx>=len(source)).any() or not np.array_equal(source[idx],target): raise ValueError('Missing exact aligned timestamp')
    return idx


def align_depth(source,target):
    idx=np.array([int(np.argmin(abs(source-d))) for d in target])
    if not np.allclose(source[idx],target,rtol=0,atol=2e-6): raise ValueError('Requested depth not present')
    if len(set(idx.tolist()))!=len(idx): raise ValueError('Duplicate target depth mapping')
    return idx

