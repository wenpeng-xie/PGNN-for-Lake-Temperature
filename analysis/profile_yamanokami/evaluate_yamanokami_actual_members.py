#!/usr/bin/env python3
"""Recompute the documented final Yamanokami ensemble on unchanged common samples.

No training, inference, checkpoint loading, candidate ranking, or member selection.
The final PGNN member IDs are fixed from the user's historical figure before scoring.
All NPZ arrays use allow_pickle=False. Archives are read without extracting paths.
Dependencies: Python 3.9+ and NumPy. evaluation_utils.py is supplied alongside.
"""
from __future__ import annotations
import argparse, io, json, re, tarfile, zipfile
from pathlib import Path
import numpy as np
from evaluation_utils import (sha256, load_npz, csv_rows, write_csv, as_time,
                              boolean, metrics, model_summary, align_time, align_depth)

FINAL_PGNN=(3,4,5,6,7,9,14,15,16,20)
ADDED=(14,15,16,20)
MODEL_NAMES={'lstm':'LSTM','rssm':'rSSM','pgnn':'PGNN'}

def zip_names(z):
    mapping={}
    for info in z.infolist():
        if info.is_dir(): continue
        key=info.filename.replace('\\','/')
        if key in mapping: raise ValueError('Duplicate normalized ZIP path: '+key)
        mapping[key]=info.filename
    return mapping

def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def run(original_ml_zip,added_zip,glm_tar,previous_review_zip,out,code_audit_zip=None):
    out.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(previous_review_zip) as z:
        prev=load_npz(z.read('evaluation_masks.npz'))
        prev_members=csv_rows(z.read('metrics_by_member.csv'))
        prev_summary=csv_rows(z.read('metrics_summary.csv'))
    time=prev['time']; depths=prev['depths']
    with tarfile.open(glm_tar,'r:gz') as t:
        names={ '/'.join(Path(m.name).parts[1:]):m for m in t if m.isfile() }
        def gr(name): return t.extractfile(names[name]).read()
        ref=load_npz(gr('reference/test_observation_reference.npz'))
        gm=load_npz(gr('reference/test_common_mask.npz'))
        selection=json.loads(gr('summary.json'))['frozen_top10']
        gsaved=csv_rows(gr('test_metrics_by_member.csv'))
        glms={r:load_npz(gr('predictions/'+r+'.npz')) for r in selection}
    if not (np.array_equal(time,ref['time']) and np.array_equal(depths,ref['depths'])):
        raise ValueError('Previous evaluation reference differs from GLM reference')
    O=ref['obs'].astype(float); G=gm['mask'].astype(bool)
    assert np.array_equal(G,prev['GLM_common'])
    histories={}
    if code_audit_zip:
        with zipfile.ZipFile(code_audit_zip) as c:
            cn=zip_names(c)
            for seed in ADDED:
                suffix=f'yamanokami_2025_lstm_structure_sgh/seed{seed}/history.json'
                found=[n for n in cn if n.endswith(suffix)]
                if len(found)!=1: raise ValueError('Ambiguous/missing history: '+suffix)
                histories[seed]=json.loads(c.read(cn[found[0]]))
    Ps=[]; audit=[]; mapping=[]; original_rep=[]; layer_rep=[]; history_rows=[]; members=[]
    with zipfile.ZipFile(original_ml_zip) as z,zipfile.ZipFile(added_zip) as added:
        an=zip_names(added)
        found_seeds=sorted(int(n.split('/')[0][4:]) for n in an if n.endswith('/lstm_overlap_avg_predictions.npz'))
        if found_seeds!=list(ADDED): raise ValueError('Unexpected added seed files: '+str(found_seeds))
        orig=csv_rows(z.read('_audit/members.csv'))
        members=[dict(r,archive='original') for r in orig if r['model']!='pgnn' or int(r['actual_seed']) in FINAL_PGNN]
        for seed in ADDED:
            members.append(dict(model='pgnn',member=f'seed{seed}',actual_seed=str(seed),archive='added',
                                prediction=f'seed{seed}/lstm_overlap_avg_predictions.npz',history=''))
        for model,expected in {'lstm':{42,*range(1,10)},'rssm':{42,*range(1,10)},'pgnn':set(FINAL_PGNN)}.items():
            actual=[int(r['actual_seed']) for r in members if r['model']==model]
            assert len(actual)==10 and set(actual)==expected,(model,actual)
        qr=csv_rows(z.read('evidence/yamanokami_2024_aligned_qc_wide.csv'))
        qc_cols=sorted((c for c in qr[0] if re.fullmatch(r'\d+(\.\d+)?\s*m',c)),key=lambda c:float(c.replace('m','')))
        qt=np.array([as_time(r['time']) for r in qr]);qd=np.array([float(c.replace('m','')) for c in qc_cols])
        ti=align_time(qt,time);di=align_depth(qd,depths);cols=[qc_cols[i] for i in di]
        rawO=np.array([[float(qr[i][c]) if qr[i][c] else np.nan for c in cols] for i in ti])
        assert np.array_equal(np.isfinite(rawO),np.isfinite(O))
        assert np.max(abs(rawO[np.isfinite(O)]-O[np.isfinite(O)]))<1e-9
        Q=np.isfinite(rawO).copy()
        for f in ('nan','range','spike','stuck'):
            Q&=~np.array([[boolean(qr[i][c+'_flag_'+f]) for c in cols] for i in ti])
        F=np.isfinite(rawO)&~np.array([[boolean(qr[i][c+'_qc_bad']) for c in cols] for i in ti])
        assert np.array_equal(Q,prev['data_only_QC'])
        for row in members:
            extra=row['archive']=='added'
            prefix=row['prediction'].rsplit('/',1)[0]
            def read_member(path): return added.read(an[path]) if extra else z.read('evidence/'+path)
            d=load_npz(read_member(row['prediction']))
            it=align_time(d['time'],time);j=align_depth(d['matched_obs_depths'],depths)
            md=d['matched_model_depths'][j];k=align_depth(d['model_depths'],md)
            p=d['pred_sel'][it[:,None],j].astype(float)
            obs=d['obs_sel'][it[:,None],j].astype(float)
            cover=d['counts'][it[:,None],k]>0
            stored=d['mask_sel'][it[:,None],j]>0
            assert np.array_equal(p,d['pred_full_model_grid'][it[:,None],k])
            assert np.array_equal(stored,F&cover)
            finite=np.isfinite(O); diff=float(np.max(abs(obs[finite]-O[finite])))
            assert diff<3e-6
            ident=dict(model=MODEL_NAMES[row['model']],member=row['member'],actual_seed=int(row['actual_seed']))
            Ps.append(dict(**ident,pred=p,cover=cover,stored_mask=stored,T0=d['T0_model'].copy()))
            for dd,mm in zip(depths,md):mapping.append(dict(**ident,obs_depth_m=float(dd),model_depth_m=float(mm),offset_m=float(mm-dd)))
            active=np.any(cover,axis=1)
            audit.append(dict(**ident,source_archive=row['archive'],source_member=row['prediction'],
                obs_max_abs_difference=diff,nonfinite_predictions=int((~np.isfinite(p)).sum()),
                first_prediction=str(time[active][0]),last_prediction=str(time[active][-1]),
                uncovered_rows_in_reference=int((~active).sum()),
                uncovered_zero_rows=int((np.all(p==0,axis=1)&~active).sum()),
                stored_mask_matches_QC_and_counts=True))
            sm=(d['mask_sel']>0)&np.isfinite(d['obs_sel'])&np.isfinite(d['pred_sel'])
            sm[:int(d['warmup_len'])]=False
            score=metrics(d['pred_sel'],d['obs_sel'],sm)
            saved=json.loads(read_member(prefix+'/metrics_overall.json'))
            delta={name:float(abs(score[name]-saved[name])) for name in ('N','RMSE','R','NSE','Bias')}
            assert delta['N']==0 and max(delta.values())<1e-5
            original_rep.append(dict(**ident,**{'recomputed_'+name:score[name] for name in ('N','RMSE','R','NSE','Bias')},
                                    **{'saved_'+name:saved[name] for name in ('N','RMSE','R','NSE','Bias')},
                                    **{'abs_difference_'+name:val for name,val in delta.items()}))
            sl=json.loads(read_member(prefix+'/metrics_by_layer.json'))
            for col,(od,mod) in enumerate(zip(d['matched_obs_depths'],d['matched_model_depths'])):
                key=f'obs_{od:.2f}m__pred_{mod:.2f}m'
                calc=metrics(d['pred_sel'][:,col],d['obs_sel'][:,col],sm[:,col])
                de={name:float(abs(calc[name]-sl[key][name])) for name in ('N','RMSE','R','NSE','Bias')}
                assert de['N']==0 and max(de.values())<1e-5
                layer_rep.append(dict(**ident,obs_depth_m=float(od),model_depth_m=float(mod),**{'abs_difference_'+name:v for name,v in de.items()}))
            h=histories.get(int(row['actual_seed'])) if extra else json.loads(z.read('evidence/'+row['history']))
            if h is not None:
                best=min(h['val'],key=lambda v:v['loss'])
                history_rows.append(dict(**ident,n_epochs=len(h['val']),best_validation_loss_epoch=int(best['epoch']),
                    best_validation_loss=float(best['loss']),best_epoch_validation_temperature_RMSE=float(best['rmse']),
                    checkpoint_identity_independently_verified=False))
    assert all(np.array_equal(Ps[0]['T0'],p['T0']) for p in Ps)
    allcover=np.logical_and.reduce([p['cover']&np.isfinite(p['pred']) for p in Ps])
    common=G&Q&allcover; fullqc=G&F&allcover; complete=common&np.all(common,axis=1)[:,None]
    for actual,key in [(common,'mask'),(fullqc,'stored_full_QC_mask'),(complete,'complete_11_mask')]:
        if not np.array_equal(actual,prev[key]): raise ValueError('Changed evaluation mask: '+key+'; do not compare silently')
    assert int(common.sum())==334433
    grec=[]
    for rid,d in glms.items():
        assert np.array_equal(d['time'],time) and np.allclose(d['depths'],depths,rtol=0,atol=1e-9)
        score=metrics(d['pred'],O,G);old=next(r for r in gsaved if r['run']==rid)
        delta={k:float(abs(score[k]-float(old[k.lower()]))) for k in ('RMSE','R','NSE','Bias')}
        assert max(delta.values())<1e-10
        grec.append(dict(member=rid,**{'abs_difference_'+k:v for k,v in delta.items()}))
        Ps.append(dict(model='GLM',member=rid,actual_seed='',pred=d['pred'].astype(float)))
    allrows=[];bydepth=[];summaries=[];change=[]
    stages={'common_data_only':common,'stored_full_QC_sensitivity':fullqc,'complete_11_depths_sensitivity':complete}
    for stage,mask in stages.items():
        sr=[]
        for p in Ps:
            ident={k:p[k] for k in ('model','member','actual_seed')}
            score=metrics(p['pred'],O,mask);record=dict(stage=stage,**ident,**score)
            sr.append(record);allrows.append(record)
            if stage=='common_data_only':
                ls=[]
                for j,d in enumerate(depths):
                    ss=metrics(p['pred'][:,j],O[:,j],mask[:,j]);ls.append(ss)
                    bydepth.append(dict(**ident,depth_m=float(d),**ss))
                assert abs(np.sqrt(sum(x['SSE'] for x in ls)/sum(x['N'] for x in ls))-score['RMSE'])<1e-10
                assert abs(np.corrcoef(p['pred'][mask],O[mask])[0,1]-score['R'])<1e-10
            prior=[r for r in prev_members if r['stage']==stage and r['model']==p['model'] and r['member']==p['member']]
            if prior:
                for k in ('N','RMSE','R','NSE','Bias'):
                    assert abs(score[k]-float(prior[0][k]))<1e-10,(p['model'],p['member'],k)
        summaries.extend(dict(stage=stage,**r) for r in model_summary(sr))
    for r in summaries:
        p=next(q for q in prev_summary if q['stage']==r['stage'] and q['model']==r['model'])
        change.append(dict(stage=r['stage'],model=r['model'],N=r['N'],old_RMSE_mean=float(p['RMSE_mean']),
            actual_members_RMSE_mean=r['RMSE_mean'],change_in_RMSE_mean=r['RMSE_mean']-float(p['RMSE_mean']),
            reason='PGNN member-set change only; evaluation positions unchanged' if r['model']=='PGNN' else 'unchanged'))
    coverage=[dict(depth_m=float(d),N=int(common[:,j].sum())) for j,d in enumerate(depths)]
    # Optional old-figure compatibility diagnostic. It is NOT the common-sample main table.
    # All finite reference observations with available finite raw predictions, including old zero placeholders.
    fmask=np.isfinite(O)&np.logical_and.reduce([np.isfinite(p['pred']) for p in Ps if p['model']=='PGNN'])
    legacy=[]
    for p in Ps:
        if p['model']=='PGNN':legacy.append(dict(model='PGNN',member=p['member'],actual_seed=p['actual_seed'],**metrics(p['pred'],O,fmask)))
    lm=np.array([r['RMSE'] for r in legacy]);lr=np.array([r['R'] for r in legacy])
    stats=dict(PGNN_RMSE_mean=float(lm.mean()),PGNN_RMSE_sd=float(lm.std(ddof=1)),PGNN_R_mean=float(lr.mean()),N=int(fmask.sum()),
        interpretation='Compatibility diagnostic using all finite reference observations and retaining zero placeholders; NOT valid final evaluation.')
    result=dict(
        sources={str(p.name):sha256(p) for p in [original_ml_zip,added_zip,glm_tar,previous_review_zip]+([code_audit_zip] if code_audit_zip else [])},
        pgnn_members=list(FINAL_PGNN),added_members=list(ADDED),n_members_per_model=10,n_common=int(common.sum()),
        evaluation_first=str(time[np.any(common,axis=1)][0]),evaluation_last=str(time[np.any(common,axis=1)][-1]),
        depths_m=depths.tolist(),mask_identical_to_previous=True,
        original_ML_metrics_reproduced=len(original_rep),original_ML_layer_metrics_reproduced=len(layer_rep),
        original_GLM_metrics_reproduced=len(grec),added_overall_metrics_reproduced=len([r for r in original_rep if r['model']=='PGNN' and r['actual_seed'] in ADDED]),
        max_original_ML_metric_abs_difference=max(max(r['abs_difference_'+k] for k in ('RMSE','R','NSE','Bias')) for r in original_rep),
        all_ML_saved_initial_profiles_identical=True,legacy_figure_compatibility=stats,
        scope='Actual historical figure member set, not original 0-9 random batch. No new test-score member selection performed in this audit.',
        limits=['Files identify members but do not establish the historical replacement criteria or an unbiased random-seed sampling protocol.',
                'No checkpoint tensors loaded; seed identity and initialization are supported by file paths and saved metadata, not inference replay.',
                'ML uses the existing nearest-depth pairing (0.05 m nominal offset); no dynamic-depth reinterpolation performed.',
                'Profile RMSE/R/NSE are not event or thermal-structure verification.',
                'The full-11-depth sensitivity uses only February-April; it does not cover the full evaluation year.'])
    write_json(out/'summary.json',result)
    for name,rows in [('metrics_summary',summaries),('metrics_by_member',allrows),('metrics_by_depth',bydepth),
                      ('prediction_audit',audit),('depth_mapping',mapping),('original_ML_metrics_reproduction',original_rep),
                      ('original_ML_layer_metrics_reproduction',layer_rep),('original_GLM_metrics_reproduction',grec),
                      ('history_audit',history_rows),('membership',members),('comparison_to_original_batch',change),
                      ('coverage_by_depth',coverage),('legacy_figure_compatibility_by_member',legacy)]:
        write_csv(out/(name+'.csv'),rows)
    np.savez_compressed(out/'evaluation_masks.npz',time=time,depths=depths,mask=common,
        stored_full_QC_mask=fullqc,complete_11_mask=complete,data_only_QC=Q,all_ML_coverage=allcover,GLM_common=G)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    for r in summaries:
        if r['stage']=='common_data_only':print(r)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-ml-zip',type=Path,required=True)
    parser.add_argument('--added-zip',type=Path,required=True)
    parser.add_argument('--glm-tar',type=Path,required=True)
    parser.add_argument('--previous-review-zip',type=Path,required=True)
    parser.add_argument('--code-audit-zip',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    run(a.original_ml_zip,a.added_zip,a.glm_tar,a.previous_review_zip,a.out,a.code_audit_zip)
