#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'  # keep text editable in SVG
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize

SITE_ORDER = ['Yamanokami', 'Shiode', 'Shin']
MODEL_ORDER = ['GLM', 'LSTM', 'rSSM', 'PGNN']
MODEL_LABELS = MODEL_ORDER


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Plot JoHRS Figure 5: layer-wise mean bias and correlation.')
    p.add_argument('--csv', type=Path, default=Path(__file__).with_name('figure5_layerwise_summary.csv'))
    p.add_argument('--outdir', type=Path, default=Path.cwd() / 'revision_figures' / 'Figure5_layerwise_v1')
    p.add_argument('--dpi', type=int, default=500)
    p.add_argument('--annotate', action='store_true', help='Write cell means inside the heatmap.')
    p.add_argument('--bias-limit', type=float, default=4.0, help='Symmetric bias color limit in °C.')
    p.add_argument('--r-min', type=float, default=0.45, help='Lower bound of correlation color scale.')
    p.add_argument('--r-max', type=float, default=1.0, help='Upper bound of correlation color scale.')
    p.add_argument('--width', type=float, default=14.4)
    p.add_argument('--height', type=float, default=6.8)
    return p.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required={'site','model','depth_m','Bias_mean','Bias_sd','R_mean','R_sd','n_members','N_min','N_max'}
    missing=required-set(df.columns)
    if missing:
        raise ValueError(f'Missing columns in {path}: {sorted(missing)}')
    for site in SITE_ORDER:
        for model in MODEL_ORDER:
            sub=df[(df.site==site)&(df.model==model)]
            if sub.empty:
                raise ValueError(f'Missing site/model rows: {site}, {model}')
            if int(sub.n_members.min()) != 10 or int(sub.n_members.max()) != 10:
                raise ValueError(f'Expected 10 frozen members for {site} {model}; found {sub.n_members.unique()}')
    return df


def centers_to_edges(vals: np.ndarray) -> np.ndarray:
    vals=np.asarray(vals,float)
    if len(vals)==1:
        return np.array([vals[0]-0.1,vals[0]+0.1])
    mids=(vals[:-1]+vals[1:])/2
    first=vals[0]-(mids[0]-vals[0])
    last=vals[-1]+(vals[-1]-mids[-1])
    return np.r_[first,mids,last]


def matrix_for(df: pd.DataFrame, site: str, field: str):
    sub=df[df.site==site].copy()
    depths=np.sort(sub.depth_m.unique().astype(float))
    mat=np.full((len(MODEL_ORDER),len(depths)),np.nan)
    sd=np.full_like(mat,np.nan)
    sd_field='Bias_sd' if field=='Bias_mean' else 'R_sd'
    for i,model in enumerate(MODEL_ORDER):
        d=sub[sub.model==model].set_index('depth_m')
        for j,z in enumerate(depths):
            if z in d.index:
                mat[i,j]=float(d.loc[z,field])
                sd[i,j]=float(d.loc[z,sd_field])
    return depths,mat,sd


def choose_text_color(value: float, metric: str, bias_limit: float, r_min: float, r_max: float) -> str:
    if metric=='Bias_mean':
        intensity=abs(value)/bias_limit
    else:
        intensity=(value-r_min)/(r_max-r_min)
    return 'white' if intensity>0.67 else 'black'


def draw_panel(ax, df, site, field, cmap, norm, annotate, args):
    depths,mat,sd=matrix_for(df,site,field)
    xedges=centers_to_edges(depths)
    yedges=np.arange(len(MODEL_ORDER)+1)-0.5
    mesh=ax.pcolormesh(xedges,yedges,mat,shading='flat',cmap=cmap,norm=norm,edgecolors='white',linewidth=0.7)
    ax.set_xlim(xedges[0],xedges[-1])
    ax.set_ylim(len(MODEL_ORDER)-0.5,-0.5)
    ax.set_xticks(depths)
    ax.set_xticklabels([f'{z:g}' for z in depths],rotation=45,ha='right',fontsize=8)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_LABELS,fontsize=9)
    ax.set_xlabel('Depth (m)')
    ax.set_title(site,fontsize=11,pad=7)
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v=mat[i,j]
                if not np.isfinite(v):
                    continue
                color=choose_text_color(v,field,args.bias_limit,args.r_min,args.r_max)
                fmt=f'{v:.2f}'
                ax.text(depths[j],i,fmt,ha='center',va='center',fontsize=6.6,color=color)
    return mesh


def main() -> None:
    args=parse_args()
    df=load_summary(args.csv)
    args.outdir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(args.width,args.height),sharey='row')
    bias_norm=TwoSlopeNorm(vmin=-args.bias_limit,vcenter=0.0,vmax=args.bias_limit)
    r_norm=Normalize(vmin=args.r_min,vmax=args.r_max)
    bias_mesh=r_mesh=None
    for col,site in enumerate(SITE_ORDER):
        bias_mesh=draw_panel(axes[0,col],df,site,'Bias_mean','RdBu_r',bias_norm,args.annotate,args)
        r_mesh=draw_panel(axes[1,col],df,site,'R_mean','viridis',r_norm,args.annotate,args)
        if col==0:
            axes[0,col].set_ylabel('(a) Mean bias (°C)\nModel')
            axes[1,col].set_ylabel('(b) Correlation (R)\nModel')
    fig.subplots_adjust(left=0.075,right=0.89,bottom=0.13,top=0.92,wspace=0.18,hspace=0.42)
    cax1=fig.add_axes([0.915,0.56,0.015,0.32])
    cbar1=fig.colorbar(bias_mesh,cax=cax1)
    cbar1.set_label('Mean bias (°C)')
    cax2=fig.add_axes([0.915,0.15,0.015,0.32])
    cbar2=fig.colorbar(r_mesh,cax=cax2)
    cbar2.set_label('Correlation (R)')
    fig.suptitle('Depth-dependent bias and correlation of temperature predictions',fontsize=13,y=0.985)
    base=args.outdir/'Figure5_layerwise_bias_correlation'
    fig.savefig(base.with_suffix('.png'),dpi=args.dpi)
    fig.savefig(base.with_suffix('.pdf'))
    fig.savefig(base.with_suffix('.svg'))
    plt.close(fig)
    print('Saved:')
    print(base.with_suffix('.png'))
    print(base.with_suffix('.pdf'))
    print(base.with_suffix('.svg'))

if __name__=='__main__':
    main()
