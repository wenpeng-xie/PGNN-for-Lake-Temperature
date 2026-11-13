#!/usr/bin/env python3
"""Read-only Shiode actual-figure-member comparison; no user code or pickle executed.
The two CSV-only PGNN members retain explicit coverage-provenance limitations.
"""
import argparse, hashlib, io, json, re, tarfile, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

MODEL={'lstm':'LSTM','rssm':'rSSM','pgnn':'PGNN'}

def file_sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def npz_bytes(b):
    with np.load(io.BytesIO(b), allow_pickle=False) as f:
        return {k: f[k] for k in f.files}

def stats(p,o,m):
    m=np.asarray(m,bool)
    if np.any(m & (~np.isfinite(p)|~np.isfinite(o))):
        raise ValueError('Nonfinite value on fixed mask')
    x=np.asarray(p,dtype=np.float64)[m];y=np.asarray(o,dtype=np.float64)[m]
    if not len(y):return dict(N=0,RMSE=np.nan,R=np.nan,NSE=np.nan,Bias=np.nan,SSE=0.)
    e=x-y; SSE=float(e@e);ym=y-y.mean();xm=x-x.mean();den=float(ym@ym)
    r=float((xm@ym)/np.sqrt((xm@xm)*den)) if den and xm@xm else np.nan
    if len(y)>1 and den and xm@xm:assert np.isclose(r,np.corrcoef(x,y)[0,1],atol=1e-12)
    return dict(N=len(y),RMSE=float(np.sqrt(SSE/len(y))),R=r,
                NSE=float(1-SSE/den) if den else np.nan,Bias=float(e.mean()),SSE=SSE)

def align(t,arr,target):
    ix=pd.Index(np.asarray(t,dtype='datetime64[ns]'))
    if not ix.is_unique or not ix.is_monotonic_increasing:raise ValueError('Invalid time index')
    inds=ix.get_indexer(target)
    if np.any(inds<0):raise ValueError('Missing target timestamps')
    return np.asarray(arr)[inds]

def boolcol(df,name):
    if df[name].isna().any():raise ValueError('Missing QC flag '+name)
    s=df[name]
    if s.dtype==bool:return s.to_numpy()
    a=s.astype(str).str.lower().map({'true':True,'false':False,'1':True,'0':False})
    if a.isna().any():raise ValueError('Invalid boolean '+name)
    return a.to_numpy(bool)

def csv_profile(df,dep,prefix=''):
    return df[[prefix+f'{d:.2f}m' for d in dep]].to_numpy(float)

def summary(rows, require_common=True):
    out=[]
    df=pd.DataFrame(rows)
    for (stage,model),g in df.groupby(['stage','model'],sort=False):
        if require_common: assert g.N.nunique()==1
        r=dict(stage=stage,model=model,n_members=len(g),N=int(g.N.iloc[0]) if g.N.nunique()==1 else None,N_min=int(g.N.min()),N_max=int(g.N.max()))
        for k in ['RMSE','R','NSE','Bias']:
            r[k+'_mean']=float(g[k].mean());r[k+'_sd']=float(g[k].std(ddof=1))
        out.append(r)
    return pd.DataFrame(out)

def run(ml_zip,glm_tar,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    with tarfile.open(glm_tar) as gt:
        gr=npz_bytes(gt.extractfile('reference/test_observation_reference.npz').read())
        gm=npz_bytes(gt.extractfile('reference/shared_test_mask.npz').read())
        time=gr['time'];dep=gr['depths'];obs=gr['obs'];base_mask=gm['mask']
        assert np.array_equal(gm['time'],time) and np.array_equal(gm['depths'],dep)
        gtable=pd.read_csv(gt.extractfile('reports/metrics_test_common.csv'))
        glm={}
        checks=[]
        for r in gtable.to_dict('records'):
            dd=npz_bytes(gt.extractfile(f"predictions/{r['run']}.npz").read())
            assert np.array_equal(dd['time'],time) and np.array_equal(dd['depths'],dep)
            glm[r['run']]=dd['pred'];s=stats(dd['pred'],obs,base_mask)
            assert s['N']==r['n_valid']
            for k in ['RMSE','R','NSE','Bias']:assert np.isclose(s[k],r[k.lower()],atol=1e-10)
            checks.append(dict(source='GLM',member=r['run'],check='reported_metrics',max_abs_diff=max(abs(s[k]-r[k.lower()]) for k in ['RMSE','R','NSE','Bias'])))
    with zipfile.ZipFile(ml_zip) as mz:
        ns=set(mz.namelist());members=json.loads(mz.read('_audit/members.json'))
        assert len(members)==30
        qc=pd.read_csv(io.BytesIO(mz.read('evidence/shiode_2025_aligned_qc_wide.csv')))
        qt=pd.to_datetime(qc['time']).to_numpy('datetime64[ns]')
        qobs=align(qt,csv_profile(qc,dep),time)
        assert np.array_equal(np.isfinite(qobs),np.isfinite(obs))
        assert np.nanmax(np.abs(qobs-obs))<1e-10
        flags={f:align(qt,np.column_stack([boolcol(qc,f'{d:.2f}m_{f}') for d in dep]),time) for f in ['flag_nan','flag_range','flag_spike','flag_stuck','flag_vert','flag_unstable','qc_bad']}
        data_valid=np.isfinite(obs)&~np.logical_or.reduce([flags[f] for f in ['flag_nan','flag_range','flag_spike','flag_stuck']])
        full_valid=np.isfinite(obs)&~flags['qc_bad']
        unified=pd.read_csv(io.BytesIO(mz.read('evidence/runs_qc/unified_plot_csv_all_layers/shiode_unified_sec_layers.csv')))
        ut=pd.to_datetime(unified['time']).to_numpy('datetime64[ns]')
        uobs=align(ut,csv_profile(unified,dep,'obs_'),time)
        assert np.array_equal(np.isfinite(uobs),np.isfinite(obs))
        assert np.nanmax(np.abs(uobs-obs))<1e-5
        models=[]; audits=[]; source_checks=[]; metric_check_rows=[]; oldrows=[]
        confirmed_coverage=np.ones(obs.shape,bool);counts_ref=None;T0s={};maprows=[]
        for row in members:
            model=MODEL[row['model']];seed=row['actual_seed'];label=row['member_label']
            files=row['included_npz'];figpath='evidence/'+row['figure_csv_candidates'][0]
            fig=pd.read_csv(io.BytesIO(mz.read(figpath)))
            ft=pd.to_datetime(fig['time']).to_numpy('datetime64[ns]')
            prefix={'lstm':'lstm_base_','rssm':'ssm_base_'}[row['model']] if seed==42 and model!='PGNN' else ''
            fp=align(ft,csv_profile(fig,dep,prefix),time)
            rec=dict(model=model,seed=seed,member=label,figure_csv=figpath)
            if files:
                assert len(files)==1
                npzpath='evidence/'+files[0];d=npz_bytes(mz.read(npzpath))
                mi=[int(np.argmin(abs(d['matched_obs_depths']-v))) for v in dep]
                assert np.allclose(d['matched_obs_depths'][mi],dep,atol=1e-6)
                mt=d['time'];pred=align(mt,d['pred_sel'][:,mi],time).astype(float)
                mo=align(mt,d['obs_sel'][:,mi],time).astype(float)
                counts=d['counts'];model_ix=[int(np.argmin(abs(d['model_depths']-v))) for v in d['matched_model_depths'][mi]]
                co=align(mt,counts[:,model_ix],time)
                coverage=co>0
                if counts_ref is None:counts_ref=co.copy()
                else: assert np.array_equal(counts_ref,co)
                confirmed_coverage &= coverage & np.isfinite(pred)
                assert np.nanmax(abs(mo-obs))<1e-5
                assert np.nanmax(abs(fp-pred))<1e-5
                rawmask=(d['mask_sel']>0)&np.isfinite(d['pred_sel'])&np.isfinite(d['obs_sel'])
                original=stats(d['pred_sel'],d['obs_sel'],rawmask)
                parent=str(Path(npzpath).parent).replace('\\','/')
                overall=parent+'/metrics_overall.json'
                if overall in ns:
                    expected=json.loads(mz.read(overall));nexp=expected.get('N',expected.get('n_valid'))
                    if nexp is not None:assert original['N']==nexp
                    errs={k:abs(original[k]-expected[k]) for k in ['RMSE','R','NSE','Bias'] if k in expected}
                    if errs:assert max(errs.values())<1e-5
                    metric_check_rows.append(dict(model=model,seed=seed,scope='overall',depth=np.nan,max_abs_diff=max(errs.values()) if errs else np.nan,N=original['N']))
                layers=parent+'/metrics_by_layer.json'
                if layers in ns:
                    lex=json.loads(mz.read(layers))
                    for j,v in enumerate(d['matched_obs_depths']):
                        pat=f'obs_{v:.2f}m__pred_{d["matched_model_depths"][j]:.2f}m'
                        key=pat if pat in lex else list(lex)[j]
                        cur=stats(d['pred_sel'][:,j],d['obs_sel'][:,j],rawmask[:,j]); exp=lex[key]
                        assert cur['N']==exp['N']
                        err=max(abs(cur[k]-exp[k]) for k in ['RMSE','R','NSE','Bias'] if k in exp)
                        assert err<1e-5
                        metric_check_rows.append(dict(model=model,seed=seed,scope='layer',depth=float(v),max_abs_diff=err,N=cur['N']))
                rec.update(source_kind='NPZ_with_counts',npz=npzpath,max_obs_difference=float(np.nanmax(abs(mo-obs))),max_csv_pred_difference=float(np.nanmax(abs(fp-pred))),first_covered_time=str(time[np.any(coverage,axis=1)][0]),last_covered_time=str(time[np.any(coverage,axis=1)][-1]),N_count_positive=int(coverage.sum()))
                T0s[f'{model}_{seed}']=d['T0_model'].tolist()
                for a,b in zip(d['matched_obs_depths'][mi],d['matched_model_depths'][mi]):maprows.append(dict(model=model,seed=seed,obs_depth=float(a),model_depth=float(b)))
            else:
                pred=fp
                rec.update(source_kind='CSV_only_NO_independent_counts',npz=None,max_obs_difference=None,max_csv_pred_difference=None,first_covered_time=None,last_covered_time=None,N_count_positive=None)
                assert model=='PGNN' and seed in (41,100)
            models.append(dict(model=model,seed=seed,member=label,pred=pred))
            audits.append(rec)
            oldmask=np.isfinite(uobs)&np.isfinite(fp)
            oldrows.append(dict(stage='legacy_figure_reproduction',model=model,member=label,seed=seed,**stats(fp,uobs,oldmask)))
        assert sorted(m['seed'] for m in models if m['model']=='PGNN')==[0,1,3,5,6,7,9,41,100,256]
        common=base_mask & data_valid & confirmed_coverage
        for m in models:common &= np.isfinite(m['pred'])
        # CSV-only members are not declared independently covered. Scores are conditional
        # on CSV values representing their predictions on the 28 verified members' support.
        full=base_mask & full_valid & confirmed_coverage
        for m in models:full &= np.isfinite(m['pred'])
        stages={'common_data_only':common,'stored_full_QC_sensitivity':full}
        complete=common & common.all(axis=1)[:,None]
        stages['complete11_profiles_sensitivity']=complete
        rows=[];layers=[]
        allmodels=[dict(model='GLM',seed=r,member=r,pred=a) for r,a in glm.items()]+models
        for stage,mask in stages.items():
            for m in allmodels:
                s=stats(m['pred'],obs,mask)
                rows.append(dict(stage=stage,model=m['model'],member=m['member'],seed=m['seed'],coverage_provenance=('conditional_for_two_CSV_only_members' if m['model']=='PGNN' and m['seed'] in (41,100) else 'counts_or_validated_GLM'),**s))
                if stage=='common_data_only':
                    ss=[]
                    for j,v in enumerate(dep):
                        e=stats(m['pred'][:,j],obs[:,j],mask[:,j]);layers.append(dict(stage=stage,model=m['model'],member=m['member'],seed=m['seed'],depth=float(v),**e));ss.append(e)
                    assert np.isclose(s['RMSE'],np.sqrt(sum(x['SSE'] for x in ss)/sum(x['N'] for x in ss)),atol=1e-12)
        brief=summary(rows); legacy=summary(oldrows, require_common=False)
        brief.to_csv(out/'metrics_summary.csv',index=False);pd.DataFrame(rows).to_csv(out/'metrics_by_member.csv',index=False)
        pd.DataFrame(layers).to_csv(out/'metrics_by_depth.csv',index=False)
        legacy.to_csv(out/'legacy_figure_reproduction_summary.csv',index=False);pd.DataFrame(oldrows).to_csv(out/'legacy_figure_reproduction_by_member.csv',index=False)
        pd.DataFrame(audits).to_csv(out/'prediction_audit.csv',index=False)
        pd.DataFrame(metric_check_rows).to_csv(out/'original_ML_metric_checks.csv',index=False)
        pd.DataFrame(checks).to_csv(out/'original_GLM_metric_checks.csv',index=False)
        pd.DataFrame(maprows).to_csv(out/'depth_mapping.csv',index=False)
        pd.DataFrame(dict(depth=dep,original_finite=base_mask.sum(axis=0),data_only_valid=(base_mask&data_valid).sum(axis=0),common=common.sum(axis=0),full_QC=full.sum(axis=0))).to_csv(out/'coverage_by_depth.csv',index=False)
        monthly=[]
        mons=time.astype('datetime64[M]')
        for mon in np.unique(mons):
            ix=mons==mon;monthly.append(dict(month=str(mon),observed_finite=int(base_mask[ix].sum()),common=int(common[ix].sum()),complete11_profiles=int(complete[ix].all(axis=1).sum())))
        pd.DataFrame(monthly).to_csv(out/'coverage_by_month.csv',index=False)
        np.savez_compressed(out/'evaluation_masks.npz',time=time,depths=dep,obs=obs,original_finite=base_mask,data_only=data_valid,confirmed_28_counts=confirmed_coverage,common=common,full_QC=full,complete11=complete)
        json.dump(T0s,open(out/'initial_profiles.json','w'),indent=2)
        interior=np.any(confirmed_coverage,axis=1)
        info=dict(n_ml=30,n_npz=28,missing_count_members=['PGNN_seed41','PGNN_seed100'],csv_only_coverage_caveat='Scores use CSV values restricted to shared positive-count coverage of the 28 independently checked members. No independent counts/T0/history for seed41 or seed100 supplied. CSV values alone do not prove valid inference coverage.',original_finite=int(base_mask.sum()),data_only_valid=int((base_mask&data_valid).sum()),common_pairs=int(common.sum()),fullQC_pairs=int(full.sum()),restored_vs_fullQC=int((common&~full_valid).sum()),QC_excluded=int((base_mask&~data_valid).sum()),coverage_excluded_afterQC=int((base_mask&data_valid&~confirmed_coverage).sum()),start=str(time[common.any(axis=1)][0]),end=str(time[common.any(axis=1)][-1]),all_28_counts_identical=True,n_overall_ML_metrics_verified=sum(r['scope']=='overall' for r in metric_check_rows),n_layer_ML_metrics_verified=sum(r['scope']=='layer' for r in metric_check_rows),NPZ_to_CSV_max_diff=max(r['max_csv_pred_difference'] or 0 for r in audits),obs_NPZ_max_diff=max(r['max_obs_difference'] or 0 for r in audits),common_mask_unknown_coverage_removed_by_finite_CSV=int((base_mask&data_valid&confirmed_coverage).sum()-common.sum()),NPZ_T0_lstm=np.asarray([v for k,v in T0s.items() if k.startswith('LSTM')]).tolist(),same_T0_within_LSTM=bool(all(np.allclose(v,T0s['LSTM_42'],atol=0) for k,v in T0s.items() if k.startswith('LSTM'))),same_T0_within_rSSM_PGNN=bool(all(np.allclose(v,T0s['rSSM_42'],atol=0) for k,v in T0s.items() if not k.startswith('LSTM'))))
        info['source_sha256']={Path(f).name:file_sha256(f) for f in [ml_zip,glm_tar]}
        json.dump(info,open(out/'summary.json','w'),indent=2)
        print(brief.to_string(index=False));print('\nLegacy reproduction\n'+legacy.to_string(index=False));print(json.dumps({k:v for k,v in info.items() if k!='NPZ_T0_lstm'},indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--ml-zip',required=True);a.add_argument('--glm-tar',required=True);a.add_argument('--out',required=True);x=a.parse_args();run(x.ml_zip,x.glm_tar,x.out)
