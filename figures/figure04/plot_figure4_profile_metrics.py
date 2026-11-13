#!/usr/bin/env python3
"""Export profile-performance panels from the finalized Table 2 source CSV.

This release exporter was created during repository assembly because the exact
standalone historical Figure 4 export cell was not recovered. It does not
recompute predictions or metrics. It uses the unrounded finalized CSV values.
The historical visual styling is not claimed to be pixel-identical.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SITES=['Yamanokami','Shiode','Shin']
MODELS=['GLM','LSTM','rSSM','PGNN']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--outdir',type=Path,default=Path('outputs/figures/figure04'))
    p.add_argument('--dpi',type=int,default=300)
    a=p.parse_args();df=pd.read_csv(a.csv)
    required={'site','model','n_members'}|{f'{m}_{v}' for m in ['RMSE','R','NSE'] for v in ['mean','sd']}
    if required-set(df):raise ValueError(f'Missing columns: {sorted(required-set(df))}')
    if df.duplicated(['site','model']).any():raise ValueError('Duplicate site/model rows.')
    if not df['n_members'].eq(10).all():raise ValueError('Expected ten model members.')
    a.outdir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.4))
    x=np.arange(3);width=.19
    for ax,metric,ylabel in zip(axes,['RMSE','R','NSE'],['RMSE (°C)','Pearson R','NSE']):
        for j,model in enumerate(MODELS):
            rows=df[df.model==model].set_index('site').reindex(SITES)
            if rows[f'{metric}_mean'].isna().any():raise ValueError(f'Missing results for {model}.')
            ax.bar(x+(j-1.5)*width,rows[f'{metric}_mean'],width,
                   yerr=rows[f'{metric}_sd'],capsize=2,label=model)
        ax.set_xticks(x,SITES);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
        if metric=='NSE':ax.axhline(0,linestyle='--',linewidth=.7)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,0,1,.92))
    for ext in ['png','pdf','svg']:fig.savefig(a.outdir/f'Figure4_overall_profile_performance.{ext}',dpi=a.dpi,bbox_inches='tight')
    plt.close(fig)
    df.to_csv(a.outdir/'Figure4_plot_data_used.csv',index=False)

if __name__=='__main__':main()
