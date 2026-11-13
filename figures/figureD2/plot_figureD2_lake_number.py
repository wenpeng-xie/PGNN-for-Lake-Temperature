#!/usr/bin/env python3
"""Export D2 from final fixed-depth Lake-number values; no physical recalculation.

The calculation is preserved in analysis/stability/. This standalone export adds
sample counts and medians to the final-data boxplot. It does not reuse the old
Figure-S notebook or claim recovery of its final annotation cell.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--outdir',type=Path,default=Path('outputs/figures/figureD2'))
    p.add_argument('--dpi',type=int,default=300)
    a=p.parse_args();df=pd.read_csv(a.csv)
    required={'site','event_id','Lake_number_peak'}
    if required-set(df):raise ValueError(f'Missing columns: {sorted(required-set(df))}')
    vals=[];rows=[]
    for site in ['yamanokami','shiode']:
        s=pd.to_numeric(df.loc[df.site.str.lower()==site,'Lake_number_peak'],errors='coerce').to_numpy(float)
        valid=s[np.isfinite(s)&(s>0)]
        if not len(valid):raise ValueError(f'No valid Lake numbers for {site}.')
        vals.append(valid);rows.append(dict(site=site,n_events=len(s),n_valid=len(valid),median=float(np.median(valid))))
    fig,ax=plt.subplots(figsize=(6.4,5.2))
    ax.boxplot(vals,tick_labels=['Yamanokami','Shiode'],showfliers=True)
    ax.set_yscale('log');ax.axhline(1,linestyle='--',linewidth=1)
    ax.set_ylabel('Lake number at observed STD(t) peak');ax.grid(axis='y',alpha=.25)
    for i,row in enumerate(rows,1):ax.text(i,.98,f'n = {row["n_valid"]}; median = {row["median"]:.1f}',
                                       transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=10)
    fig.tight_layout();a.outdir.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(a.outdir/f'Figure_D2_event_lake_number_final.{ext}',dpi=a.dpi,bbox_inches='tight')
    plt.close(fig);pd.DataFrame(rows).to_csv(a.outdir/'Figure_D2_plot_audit.csv',index=False)

if __name__=='__main__':main()
