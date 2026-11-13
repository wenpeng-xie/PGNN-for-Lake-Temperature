#!/usr/bin/env python3
"""Read-only review of Shin retraining returns and a frozen GLM evaluation.
No training, no checkpoint selection based on test data, no interpolation.
Dependencies: Python >=3.10, numpy. Input directories are safely extracted archives.
"""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np

def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write_json(p,v):
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def write_csv(p,rows):
    rows=list(rows)
    if not rows: return
    with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def score(pred,obs,mask):
    assert pred.shape == obs.shape == mask.shape
    assert np.isfinite(pred[mask]).all() and np.isfinite(obs[mask]).all()
    p=np.asarray(pred[mask],float);o=np.asarray(obs[mask],float)
    if not len(p): raise ValueError('No common samples')
    e=p-o
    mse=np.mean(e*e); bias=np.mean(e)
    return dict(N=int(len(p)),RMSE=float(np.sqrt(mse)),R=float(np.corrcoef(p,o)[0,1]),
                NSE=float(1-np.sum(e*e)/np.sum((o-o.mean())**2)),Bias=float(bias),
                centered_RMSE_diagnostic=float(np.sqrt(np.mean((e-bias)**2))),
                squared_bias_fraction_of_MSE=float(bias*bias/mse))

def aggregate(rows):
    result=[]
    for model in dict.fromkeys(r['model'] for r in rows):
        sub=[r for r in rows if r['model']==model]
        d=dict(model=model,n_members=len(sub),N_min=min(r['N'] for r in sub),N_max=max(r['N'] for r in sub))
        for k in ['RMSE','R','NSE','Bias','centered_RMSE_diagnostic','squared_bias_fraction_of_MSE']:
            v=np.array([r[k] for r in sub]);d[k+'_mean']=float(v.mean());d[k+'_sd']=float(v.std(ddof=1))
        result.append(d)
    return result

def run(ml,glm,code,out):
    out.mkdir(parents=True,exist_ok=True)
    manifest=load_json(ml/'run_manifest.json')
    h=hashlib.sha256()
    for p in sorted(code.glob('*.py')):h.update(p.name.encode());h.update(p.read_bytes())
    code_hash=h.hexdigest();assert code_hash==manifest['implementation_sha256']
    assert load_json(ml/'STATUS.json')['complete']
    assert load_json(ml/'ALL_30_MODELS_FROZEN.json')['frozen_before_test']
    audit=[];ml_members=[];reported_diffs=[];all18_diffs=[];csv_diffs=[]
    reference=None;refreport=None
    expected={'lstm':[42,*range(1,10)],'rssm':[42,*range(1,10)],'pgnn':list(range(10))}
    for model in ['lstm','rssm','pgnn']:
        mr=ml/'models'/model; report=load_json(mr/'data_split_report.json')
        if refreport is None: refreport=report
        else: assert report==refreport
        assert report['source_hash']==code_hash
        assert report['test_used_for_training_statistics'] is False
        cfg=manifest['configs'][model];assert cfg['actual_seeds']==expected[model]
        assert cfg['development_ratio']==.7 and cfg['internal_train_ratio']==.8 and cfg['epochs']==200
        dirs=sorted(mr.glob('member_*_seed*'));assert len(dirs)==10
        frozen=load_json(mr/'frozen_members.json');assert len(frozen)==10
        for idx,d in enumerate(dirs):
            conf=load_json(d/'config.json');done=load_json(d/'TRAINING_DONE.json');hist=load_json(d/'history.json')
            assert conf['config']==cfg and conf['member_id']==idx and conf['actual_seed']==expected[model][idx]
            assert conf['implementation_sha256']==code_hash and conf['data_sha256']==manifest['data_sha256']
            assert done==frozen[idx] and done['fingerprint']==conf['fingerprint']
            assert list(hist)==['train','val'] and done['completed_epochs']==200
            for split in ['train','val']:
                assert [r['epoch'] for r in hist[split]]==list(range(1,201))
                assert all(np.isfinite(r['loss']) and np.isfinite(r['rmse']) for r in hist[split])
            best=min(hist['val'],key=lambda r:r['loss'])
            assert best['epoch']==done['best_epoch'] and abs(best['loss']-done['best_val_loss'])<1e-10
            assert all(r['warmup_obs']==0 for r in hist['val'])
            lossdiff=[]
            for split in ['train','val']:
                for row in hist[split]:
                    calc=row['obs']+cfg['time_weight']*row['t_smooth']+cfg['depth_weight']*row['z_smooth']+cfg['range_weight']*row['range']+cfg['std_weight']*row['structure']+cfg['gmean_weight']*row['dstructure']
                    lossdiff.append(abs(calc-row['loss']))
            with np.load(d/'test_predictions.npz',allow_pickle=False) as z: a={k:z[k] for k in z.files}
            assert a['actual_seed']==expected[model][idx] and a['member_id']==idx
            assert a['time'].shape==(report['test']['n_rows'],) and a['pred'].shape==(len(a['time']),18)
            assert np.isfinite(a['pred']).all() and a['prediction_coverage'].all() and (a['count']>0).all()
            assert not np.any(np.all(a['pred']==0,axis=1))
            assert np.array_equal(a['evaluation_mask'],a['obs_valid'] & a['prediction_coverage'] & np.isfinite(a['pred']))
            assert np.array_equal(np.isfinite(a['obs']),a['obs_valid'])
            if reference is None:reference=a
            else:
                for k in ['time','depths','obs_valid','evaluation_mask','count','T0']:
                    assert np.array_equal(reference[k],a[k]),(model,idx,k)
                assert np.array_equal(reference['obs'],a['obs'],equal_nan=True)
            paper=a['depths']<=3.450001
            q=score(a['pred'][:,paper],a['obs'][:,paper],a['evaluation_mask'][:,paper])
            reported=load_json(d/'test_metrics.json')
            for k in ['N','RMSE','R','NSE','Bias']:
                reported_diffs.append(abs(q[k]-reported['paper_17_channels'][k]))
            q18=score(a['pred'],a['obs'],a['evaluation_mask'])
            for k in ['N','RMSE','R','NSE','Bias']:
                all18_diffs.append(abs(q18[k]-reported['all_18_channels'][k]))
            best_train=hist['train'][done['best_epoch']-1]
            audit.append(dict(model=model,member_id=idx,actual_seed=int(a['actual_seed']),epochs=200,
                best_epoch=done['best_epoch'],best_val_loss=best['loss'],best_val_RMSE=best['rmse'],
                train_RMSE_at_selected_epoch=best_train['rmse'],last_val_loss=hist['val'][-1]['loss'],
                last_train_loss=hist['train'][-1]['loss'],loss_formula_max_error=max(lossdiff),
                predictions_all_finite=True,predictions_all_covered=True,no_zero_profiles=True,
                normalization_hash=hashlib.sha256((mr/'normalization_and_T0.npz').read_bytes()).hexdigest(),
                selected_checkpoint_sha256=done['selected_checkpoint_sha256']))
            ml_members.append(dict(model=model.upper() if model!='rssm' else 'rSSM',member_id=idx,actual_seed=int(a['actual_seed']),
                                   best_epoch=done['best_epoch'],pred=a['pred'][:,paper].astype(float)))
    with np.load(glm/'test_predictions.npz',allow_pickle=False) as z:g={k:z[k] for k in z.files}
    assert np.array_equal(reference['time'],g['time'])
    paper=reference['depths']<=3.450001
    assert np.allclose(reference['depths'][paper],g['depths'],rtol=0,atol=1e-8)
    valid=reference['obs_valid'][:,paper]
    obs=g['obs']
    diff=float(np.max(np.abs(obs[valid]-reference['obs'][:,paper][valid])))
    assert diff<2e-6
    common=g['common_mask'] & valid
    for m in ml_members:common &= np.isfinite(m['pred'])
    glm_index=[int(np.where(g['run_ids']==rid)[0][0]) for rid in g['new_top10']]
    assert len(glm_index)==10
    all_members=[dict(model='GLM',member_id=i,actual_seed='',best_epoch='',run_id=str(g['run_ids'][j]),pred=g['pred'][j]) for i,j in enumerate(glm_index)]
    for m in ml_members:m['run_id']=''
    all_members.extend(ml_members)
    common_rows=[];layer_rows=[];month_rows=[];fixed_rows=[]
    fixed=(g['depths']<=3.250001)
    # Supplemental fixed 16-channel complete-profile evaluation. No time-gap interpolation.
    fixed_times=common[:,fixed].all(axis=1)
    fixed_mask=np.zeros_like(common);fixed_mask[:,fixed]=fixed_times[:,None]
    for m in all_members:
        ident={k:m[k] for k in ['model','member_id','actual_seed','best_epoch','run_id']}
        common_rows.append(dict(**ident,**score(m['pred'],obs,common)))
        fixed_rows.append(dict(**ident,**score(m['pred'],obs,fixed_mask)))
        for j,z in enumerate(g['depths']):
            layer_rows.append(dict(**ident,depth_m=float(z),**score(m['pred'][:,j],obs[:,j],common[:,j])))
        for month in np.unique(g['time'].astype('datetime64[M]')):
            mask=common & (g['time'].astype('datetime64[M]')==month)[:,None]
            month_rows.append(dict(**ident,month=str(month),**score(m['pred'],obs,mask)))
    coverage=[dict(depth_m=float(z),glm_previous_common=int(g['common_mask'][:,j].sum()),
                   ml_qc_valid=int(valid[:,j].sum()),new_common=int(common[:,j].sum()),
                   removed_from_old_common=int((g['common_mask'][:,j]&~common[:,j]).sum())) for j,z in enumerate(g['depths'])]
    summary=aggregate(common_rows)
    write_csv(out/'common_metrics_by_member.csv',common_rows);write_csv(out/'common_metrics_summary.csv',summary)
    write_csv(out/'common_metrics_by_depth.csv',layer_rows);write_csv(out/'common_metrics_by_month.csv',month_rows)
    write_csv(out/'fixed16_complete_profiles_by_member.csv',fixed_rows);write_csv(out/'fixed16_complete_profiles_summary.csv',aggregate(fixed_rows))
    write_csv(out/'training_member_audit.csv',audit);write_csv(out/'coverage_by_depth.csv',coverage)
    np.savez_compressed(out/'common_evaluation_mask.npz',time=g['time'],depths=g['depths'],mask=common,
                        mask_16_complete=fixed_mask,ml_obs_valid=valid,glm_previous_common_mask=g['common_mask'])
    # Test multiple independently evaluated identities; no claim of reproducing GPU model training.
    per_member_reconstruction=[]
    for m,r in zip(all_members,common_rows):
        ll=[x for x in layer_rows if x['model']==m['model'] and x['member_id']==m['member_id']]
        reconstruction=np.sqrt(sum(x['N']*x['RMSE']**2 for x in ll)/r['N'])
        per_member_reconstruction.append(abs(reconstruction-r['RMSE']))
    best_summary={}
    for model in expected:
        rr=[r for r in audit if r['model']==model]
        bb=[r['best_epoch'] for r in rr]
        best_summary[model]=dict(min=min(bb),max=max(bb),at_or_after_190=sum(x>=190 for x in bb),at_200=sum(x==200 for x in bb),
                                 median=float(np.median(bb)))
    observed=obs[common]
    checks=dict(status='ok',completed_members=30,epochs_per_member=200,epochs_history_records=12000,
        actual_seeds=expected,implementation_sha256=code_hash,implementation_hash_matches_submitted_package=True,
        data_sha256=manifest['data_sha256'],all_three_split_reports_identical=True,
        selected_epochs_match_internal_validation_argmin=True,validation_warmup_labels_excluded=True,
        maximum_returned17_metric_recalculation_error=max(reported_diffs),
        maximum_returned18_metric_recalculation_error=max(all18_diffs),
        maximum_rmse_reconstruction_from_layers_error=max(per_member_reconstruction),
        observation_max_absolute_difference_C=diff,model_depths=18,evaluation_depths=17,
        n_test_times=len(g['time']),n_previous_GLM_common_pairs=int(g['common_mask'].sum()),
        n_ml17_valid_pairs=int(valid.sum()),n_new_common_pairs=int(common.sum()),
        n_removed_from_old_common=int((g['common_mask']&~common).sum()),
        n_new_common_times=int(common.any(axis=1).sum()),
        first_common_time=str(g['time'][common.any(axis=1)][0]),last_common_time=str(g['time'][common.any(axis=1)][-1]),
        fixed16_complete_profile_times=int(fixed_times.sum()),fixed16_pairs=int(fixed_mask.sum()),
        common_obs_mean_C=float(observed.mean()),common_obs_std_C=float(observed.std()),
        coverage_all_test_targets=True,no_terminal_zero_profiles=True,
        best_epochs=best_summary,
        note='Read-only source/records/predictions audit, not checkpoint-forward or bitwise training reproduction. '
             'Retrospective correction of previously inspected test period; not a newly unseen independent dataset. '
             'GLM spread is calibrated-parameter-set spread; ML spread is across fixed random seeds. '
             'Same score mask does not imply identical training-data budgets: ML optimizes on first56%, GLM calibration on development70%.')
    write_json(out/'checks.json',checks);write_json(out/'split_protocol.json',refreport)
    print(json.dumps(checks,ensure_ascii=False,indent=2));print('\nCOMMON SUMMARY\n'+json.dumps(summary,indent=2))
    return checks,summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--ml',type=Path,required=True);ap.add_argument('--glm',type=Path,required=True)
    ap.add_argument('--code',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    run(a.ml,a.glm,a.code,a.out)
