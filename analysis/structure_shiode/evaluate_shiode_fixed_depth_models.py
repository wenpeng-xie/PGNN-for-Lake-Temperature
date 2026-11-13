#!/usr/bin/env python3
"""Shiode R1-5 step 1B: compare existing predictions on step-1A event catalogs.

NO training, inference, GLM simulation, parameter selection, or event re-detection.
Python >=3.9; numpy and pandas. Archives are read without extraction or unpickling.
The primary event comparator preserves the historical ensemble-temperature-mean
and within-window max-minus-min definition; per-member diagnostics are separate.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import sys
import tarfile
import traceback
import warnings
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = '1.0.0'
BASE = Path(__file__).resolve().parent
DEPTHS = np.array([25 + 20*j for j in range(11)], dtype=float)/100
MODEL_NAMES = ('GLM', 'LSTM', 'rSSM', 'PGNN')
EXPECTED_ML_SEEDS = {
    'LSTM': [1,2,3,4,5,6,7,8,9,42],
    'rSSM': [1,2,3,4,5,6,7,8,9,42],
    'PGNN': [0,1,3,5,6,7,9,41,100,256],
}
EXPECTED_GLM_RUNS = ['run_832','run_801','run_956','run_946','run_962',
                     'run_334','run_893','run_679','run_483','run_311']
STEP_NS = 10*60*1_000_000_000
WINDOW_STEPS = 18
MIN_LAYERS_CONTINUOUS = 8


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def safe_json(value):
    if isinstance(value, dict): return {str(k):safe_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [safe_json(v) for v in value]
    if isinstance(value,np.ndarray): return safe_json(value.tolist())
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(np.bool_,)): return bool(value)
    if isinstance(value,(np.floating,float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,Path): return str(value)
    return value


def write_json(path,value):
    Path(path).write_text(json.dumps(safe_json(value),ensure_ascii=False,
                         indent=2,allow_nan=False),encoding='utf-8')


def save_csv(path,rows):
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path,index=False,encoding='utf-8-sig')


def read_npz(blob):
    with np.load(io.BytesIO(blob),allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


class Archive:
    """Read member bytes, never extract paths and never execute archive code."""
    def __init__(self,path):
        self.path=Path(path)
        if zipfile.is_zipfile(self.path):
            self.handle=zipfile.ZipFile(self.path)
            names=self.handle.namelist()
            self.kind='zip'
        else:
            self.handle=tarfile.open(self.path,'r:*')
            names=[m.name for m in self.handle.getmembers() if m.isfile()]
            self.kind='tar'
        require(len(names)==len(set(names)),f'Duplicate archive paths: {path}')
        self.names=set(names)
    def __enter__(self): return self
    def __exit__(self,*args): self.handle.close()
    def resolve(self,name):
        if name in self.names: return name
        candidates=[p for p in self.names if p.endswith('/'+name)]
        require(len(candidates)==1,f'Cannot uniquely locate {name} in {self.path.name}')
        return candidates[0]
    def read(self,name):
        name=self.resolve(name)
        if self.kind=='zip': return self.handle.read(name)
        f=self.handle.extractfile(name)
        require(f is not None,f'Not a regular archive member: {name}')
        return f.read()
    def json(self,name): return json.loads(self.read(name).decode('utf-8-sig'))
    def csv(self,name): return pd.read_csv(io.BytesIO(self.read(name)))
    def npz(self,name): return read_npz(self.read(name))


def times(values):
    ix=pd.DatetimeIndex(pd.to_datetime(values,errors='raise'))
    require(ix.tz is None,'Timezone-aware input needs explicit review; no implicit conversion.')
    require(not ix.hasnans and ix.is_unique and ix.is_monotonic_increasing,
            'Invalid, duplicate, or unsorted timestamps.')
    return ix.to_numpy(dtype='datetime64[ns]')


def align(source_time,values,target,fill=np.nan):
    st=times(source_time)
    values=np.asarray(values)
    require(len(st)==len(values),'Time and values have different lengths.')
    ix=pd.Index(st).get_indexer(target)
    out=np.full((len(target),)+values.shape[1:],fill,dtype=float)
    present=ix>=0
    out[present]=values[ix[present]]
    return out,present


def depth_indices(source,target=DEPTHS):
    source=np.asarray(source,dtype=float)
    require(source.ndim==1 and len(source)>0 and np.isfinite(source).all(),
            'Invalid depth array.')
    idx=[int(np.argmin(abs(source-v))) for v in target]
    require(len(set(idx))==len(idx) and np.allclose(source[idx],target,rtol=0,atol=1e-5),
            'Cannot exactly match nominal depth channels; no automatic remapping.')
    return idx


def same_observation(reference,other,name,tolerance=1e-5):
    use=np.isfinite(reference)
    require(np.isfinite(other[use]).all(),f'{name}: nonfinite observed values at reference cells.')
    diff=float(np.max(abs(reference[use]-other[use]))) if use.any() else 0.0
    require(diff<=tolerance,f'{name}: observation mismatch, max diff={diff}')
    return diff


def row_std(T):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanstd(T,axis=1,ddof=0)


def structure(T,observed_valid,complete_only=False):
    """Same observed channels for models and observations; no depth interpolation here."""
    mask=np.asarray(observed_valid,bool)
    T=np.asarray(T,float)
    count=mask.sum(axis=1)
    finite_required=np.all(~mask|np.isfinite(T),axis=1)
    valid=finite_required & (count==len(DEPTHS) if complete_only
                              else count>=MIN_LAYERS_CONTINUOUS)
    masked=np.where(mask,T,np.nan)
    sd=row_std(masked)
    pair=mask[:,1:] & mask[:,:-1]
    grad=abs(np.diff(T,axis=1)/np.diff(DEPTHS)[None,:])
    den=pair.sum(axis=1)
    g=np.full(len(T),np.nan)
    good=valid & (den>0)
    numerator=np.where(pair,grad,0).sum(axis=1)
    g[good]=numerator[good]/den[good]
    sd[~valid]=np.nan
    return sd,g


def legacy_gmean(T,mask,allowed_rows):
    """Historical comparator only: >=70% channels; interpolate including edge extension."""
    out=np.full(len(T),np.nan)
    for i in np.flatnonzero(allowed_rows):
        good=mask[i]
        if good.sum()>=MIN_LAYERS_CONTINUOUS and np.isfinite(T[i,good]).all():
            filled=np.interp(DEPTHS,DEPTHS[good],T[i,good])
            out[i]=np.mean(abs(np.diff(filled)/np.diff(DEPTHS)))
    return out


def series_metrics(pred,obs,allowed):
    m=np.asarray(allowed,bool)
    require(np.isfinite(pred[m]).all() and np.isfinite(obs[m]).all(),
            'Nonfinite structure metric on common scoring times.')
    p=np.asarray(pred,dtype=float)[m];o=np.asarray(obs,dtype=float)[m]
    if not len(o): return dict(N_time=0,RMSE=np.nan,MAE=np.nan,Bias=np.nan,R=np.nan,VR=np.nan)
    e=p-o;oc=o-o.mean();pc=p-p.mean()
    os=float(np.std(o,ddof=0));ps=float(np.std(p,ddof=0))
    r=float((pc@oc)/np.sqrt((pc@pc)*(oc@oc))) if os>0 and ps>0 else np.nan
    return dict(N_time=len(o),RMSE=float(np.sqrt(np.mean(e*e))),
                MAE=float(np.mean(abs(e))),Bias=float(e.mean()),R=r,
                VR=ps/os if os>0 else np.nan)


@dataclass
class Member:
    model:str
    member:str
    pred:np.ndarray
    valid:np.ndarray
    metadata:dict


def load_observations(path):
    with Archive(path) as a:
        s=a.json('summary.json')
        d=a.npz('fixed_depth_observation_reference.npz')
        catalogs={
            'A_quality_available':a.csv('events_A_data_quality_available.csv'),
            'B_fixed11_complete3h':a.csv('events_B_fixed11_complete3h.csv'),
        }
        audit_A=a.csv('audit_A_event_windows.csv')
        coverage_month=a.csv('coverage_by_month.csv')
    t=times(d['time'])
    require(np.all(np.diff(t.astype(np.int64))==STEP_NS),'Step1A time grid must be regular 10 minutes.')
    require(np.allclose(d['depths'],DEPTHS,rtol=0,atol=1e-8),'Step1A depth mismatch.')
    require(s['event_window_minutes']==180 and s['STD_ddof']==0,'Step1A protocol mismatch.')
    raw=np.asarray(d['obs_raw'],float);q=np.asarray(d['valid_mask'],bool)
    require(raw.shape==(len(t),11) and q.shape==raw.shape,'Step1A array shape mismatch.')
    require(np.isfinite(raw[q]).all(),'Valid observations cannot contain nonfinite values.')
    clean=np.where(q,raw,np.nan)
    require(np.allclose(clean,d['obs_data_quality'],equal_nan=True),'Step1A clean-observation mismatch.')
    require(np.array_equal(q.all(axis=1),d['strict_complete_profile']),'Step1A completeness mismatch.')
    for key,df in catalogs.items():
        require(df.event_id.is_unique,'Duplicate event IDs.')
        for r in df.itertuples(index=False):
            i,j,p,k=map(int,(r.start_index,r.end_index,r.peak_index,r.min_index))
            require(0<=i<=p<k<=j<len(t),'Invalid stored event indices.')
            require(j-i==WINDOW_STEPS,'Require complete 3h catalog windows; no tail truncation.')
            require(pd.Timestamp(r.start_time)==pd.Timestamp(t[i]) and
                    pd.Timestamp(r.end_time)==pd.Timestamp(t[j]),'Event dates do not match array indices.')
            if key.startswith('B_'):require(q[i:j+1].all(),'B catalog contains an incomplete window.')
    require(np.array_equal(catalogs['B_fixed11_complete3h'].start_index.to_numpy(int),
                           d['strict_event_start_indices']), 'B catalog/reference mismatch.')
    return s,d,t,raw,q,clean,catalogs,audit_A,coverage_month


def load_glm(path,t,raw):
    records=[];items=[]
    with Archive(path) as a:
        ref=a.npz('reference/test_observation_reference.npz')
        di=depth_indices(ref['depths'])
        ob,present=align(ref['time'],ref['obs'][:,di],t)
        require(present.all(),'GLM reference does not span Step1A time grid.')
        diff=same_observation(raw,ob,'GLM observation reference')
        table=a.csv('reports/metrics_test_common.csv').sort_values('calibration_rank')
        require(table.run.tolist()==EXPECTED_GLM_RUNS,'Unexpected GLM member set/order: frozen selection changed.')
        for r in table.itertuples(index=False):
            name=f'predictions/{r.run}.npz';d=a.npz(name)
            di=depth_indices(d['depths'])
            p,pr=align(d['time'],d['pred'][:,di],t)
            v=np.isfinite(p)&pr[:,None]
            meta=dict(model='GLM',member=r.run,source=name,
                      coverage_evidence='corrected_extraction_finite_inside_water_column',
                      independent_counts=True,missing_target_timestamps=int((~pr).sum()),
                      original_calibration_rank=int(r.calibration_rank),max_observation_difference=diff)
            items.append(Member('GLM',r.run,p,v,meta));records.append(meta)
    return items,records


def load_ml(path,t,raw,obs_hash,require_verified):
    items=[];records=[];maps=[]
    names={'lstm':'LSTM','rssm':'rSSM','pgnn':'PGNN'}
    with Archive(path) as a:
        rows=a.json('_audit/members.json')
        require(len(rows)==30,'Expected exactly 30 ML members; no silent member removal.')
        qcsv=a.read('evidence/shiode_2025_aligned_qc_wide.csv')
        require(hashlib.sha256(qcsv).hexdigest()==obs_hash,
                'Step1A and ML archive use different raw QC tables. Stop and reconcile versions.')
        for row in rows:
            model=names[row['model']];seed=int(row['actual_seed']);label=f'seed{seed}'
            npzs=row['included_npz']
            meta=dict(model=model,member=label,actual_seed=seed,
                      original_label=row['member_label'])
            if npzs:
                require(len(npzs)==1,'Ambiguous NPZ prediction source.')
                name='evidence/'+npzs[0];d=a.npz(name)
                di=depth_indices(d['matched_obs_depths'])
                p,pr=align(d['time'],d['pred_sel'][:,di],t)
                ob,opr=align(d['time'],d['obs_sel'][:,di],t)
                require(opr.all(),'ML observation array does not span Step1A grid.')
                diff=same_observation(raw,ob,f'{model} {label}')
                modeldi=depth_indices(d['model_depths'],np.asarray(d['matched_model_depths'])[di])
                require(np.asarray(d['counts']).ndim==2,'Unexpected counts layout; do not infer coverage.')
                co,cpr=align(d['time'],d['counts'][:,modeldi],t,fill=0.)
                require(np.isfinite(co).all() and (co>=0).all(),'Invalid prediction counts.')
                v=(co>0)&np.isfinite(p)&pr[:,None]&cpr[:,None]
                meta.update(source=name,coverage_evidence='NPZ_counts_positive',independent_counts=True,
                            missing_target_timestamps=int((~pr).sum()),max_observation_difference=diff)
                for od,md in zip(np.asarray(d['matched_obs_depths'])[di],np.asarray(d['matched_model_depths'])[di]):
                    maps.append(dict(model=model,member=label,observed_nominal_depth_m=float(od),
                                     prediction_nominal_depth_m=float(md),mapping='existing_NPZ_mapping_not_new_interpolation'))
            else:
                require(model=='PGNN' and seed in (41,100),'Unexpected CSV-only member.')
                require(not require_verified,'Independent coverage record missing; strict evidence mode requested.')
                paths=row['figure_csv_candidates'];require(len(paths)==1,'Ambiguous CSV source.')
                name='evidence/'+paths[0];df=a.csv(name)
                cols=[f'{z:.2f}m' for z in DEPTHS]
                require(all(c in df for c in ['time']+cols),'Missing CSV temperature/time columns.')
                p,pr=align(df['time'],df[cols].to_numpy(float),t)
                v=np.isfinite(p)&pr[:,None]
                meta.update(source=name,coverage_evidence='CSV_finite_only_CONDITIONAL_no_independent_counts',
                            independent_counts=False,missing_target_timestamps=int((~pr).sum()))
                for od in DEPTHS:
                    maps.append(dict(model=model,member=label,observed_nominal_depth_m=float(od),
                                     prediction_nominal_depth_m=None,mapping='CSV_label_only_mapping_provenance_unverified'))
            items.append(Member(model,label,p,v,meta));records.append(meta)
    for model,expected in EXPECTED_ML_SEEDS.items():
        actual=sorted(int(m.metadata['actual_seed']) for m in items if m.model==model)
        require(actual==expected,f'{model}: member set differs from frozen actual-figure set.')
    require(len({(m.model,m.member) for m in items})==30,'Duplicate ML member.')
    return items,records,maps


def range_values(series,i,j):
    w=np.asarray(series[i:j+1],float)
    require(np.isfinite(w).all(),'Nonfinite structure within accepted event window.')
    peak=i+int(np.argmax(w));trough=i+int(np.argmin(w))
    return float(w.max()-w.min()),peak,trough


def event_record(series,obs_series,r,model,member,catalog,aggregation,t):
    i,j,p,k=map(int,(r.start_index,r.end_index,r.peak_index,r.min_index))
    od,op,oq=range_values(obs_series,i,j)
    require(np.isclose(od,float(r.drop_degC),atol=1e-8,rtol=1e-8), 'Stored observation event drop mismatch.')
    require(od>0,'Observed event amplitude must be positive.')
    md,mp,mq=range_values(series,i,j)
    ratio=md/od
    signed=float(series[p]-series[k])
    return dict(catalog=catalog,aggregation=aggregation,model=model,member=member,
                event_id=r.event_id,start_time=str(t[i]),end_time=str(t[j]),
                obs_peak_time=str(t[p]),obs_min_time=str(t[k]),obs_drop_degC=od,
                model_window_range_degC=md,drop_ratio=ratio,absolute_ratio_error=abs(ratio-1),
                absolute_amplitude_error_degC=abs(md-od),model_peak_time=str(t[mp]),
                model_min_time=str(t[mq]),model_min_after_max=bool(mq>mp),
                model_signed_change_at_observed_extrema_degC=signed,
                signed_change_at_observed_extrema_ratio=signed/od)


def event_summary(frame,aggregation):
    rows=[]
    if frame.empty:return pd.DataFrame()
    for (cat,model,member),g in frame.groupby(['catalog','model','member'],sort=False):
        r=dict(catalog=cat,aggregation=aggregation,model=model,member=member,N_events=len(g))
        for name in ['drop_ratio','absolute_ratio_error','absolute_amplitude_error_degC',
                     'model_signed_change_at_observed_extrema_degC']:
            r[name+'_mean']=float(g[name].mean());r[name+'_median']=float(g[name].median())
        r['drop_ratio_q25']=float(g.drop_ratio.quantile(.25))
        r['drop_ratio_q75']=float(g.drop_ratio.quantile(.75))
        r['fraction_ratio_below_one']=float((g.drop_ratio<1).mean())
        r['fraction_model_min_after_max']=float(g.model_min_after_max.mean())
        rows.append(r)
    return pd.DataFrame(rows)


def aggregate_member_summaries(frame,group_keys,metrics,count_key):
    rows=[]
    if frame.empty:return pd.DataFrame()
    for key,g in frame.groupby(group_keys,sort=False):
        if not isinstance(key,tuple):key=(key,)
        r=dict(zip(group_keys,key));r['n_members']=len(g)
        require(g[count_key].nunique()==1,'Members have different evaluation sample counts.')
        r[count_key]=int(g[count_key].iloc[0])
        for k in metrics:
            r[k+'_mean_across_members']=float(g[k].mean())
            r[k+'_sd_across_members']=float(g[k].std(ddof=1))
        rows.append(r)
    return pd.DataFrame(rows)


def report_text(summary,event_sum,structure_sum):
    lines=['# Shiode R1-5 第1B步：现有四模型固定深度／事件对照',
           '', '本报告由用户本地运行脚本生成。未训练、未推理、未运行GLM、未重新挑选成员或事件阈值。',
           '', '## 输入与共同样本',
           f'- 时间轴：{summary["time_start"]}—{summary["time_end"]}，10分钟，固定11个名义观测深度。',
           '- 模型通道采用既有NPZ深度映射，不新增温度深度插值；ML实际名义深度与观测可有0.05 m偏移。',
           f'- A可用层连续诊断时刻：{summary["continuous_A_time_count"]}；B完整11层时刻：{summary["continuous_B_time_count"]}。',
           f'- 输入观测事件 A={summary["catalog_A_input_count"]}、B={summary["catalog_B_input_count"]}。',
           f'- 四模型共同可评价事件 A={summary["catalog_A_common_count"]}、B={summary["catalog_B_common_count"]}。',
           '- 不将四模型缺预测窗口静默删除：原因见event_eligibility.csv。相同目录下所有成员使用同一事件集合。',
           '', '## 主要事件对照：先平均10个成员温度，再计算STD',
           '保留原图的集合处理方式；不是先计算成员指标再平均。下表drop ratio是同一观测窗口内模型STD极差／观测STD极差。',
           '模型极小值是否晚于极大值另列；极差之比不等于事件命中率或时间预测精度。',
           '', '|目录|模型|共同事件数|drop ratio均值|中位数|平均绝对幅度误差°C|',
           '|---|---|---:|---:|---:|---:|']
    for r in event_sum.to_dict('records'):
        lines.append(f'|{r["catalog"]}|{r["model"]}|{r["N_events"]}|{r["drop_ratio_mean"]:.6f}|{r["drop_ratio_median"]:.6f}|{r["absolute_amplitude_error_degC_mean"]:.6f}|')
    lines+=['','## 应如何比较',
            '1. 先看A与B的共同事件数、事件幅度与绝对误差；不预设排名必须保持。drop ratio越接近1越准确，不是越大越好。',
            '2. per_member表是逐成员计算后汇总，与集合平均温度先计算的结果分开保存，不能把10个成员×事件数当作独立观测样本数。',
            '3. continuous结构表同时给出A的相邻有效层版本、旧深度插值Gmean对照及B完整11层版本。旧插值仅用于对照，不进入严格事件。',
            '4. 严格观测事件清单不重新识别；任何未纳入模型比较的事件单列原因，不能称为假事件。',
            '5. 完整深度筛选的月份覆盖见step1A_coverage_by_month.csv及common_coverage_by_month.csv。没有完整窗口的月份不能解释为没有事件。',
            '', '## 限制与下一步',
            '- 两个CSV-only成员没有独立counts。本次沿用既有评价约定：仅在其余成员已验证覆盖范围使用其有限CSV值。涉及其的PGNN与全40成员共同样本结果具有此条件；未伪造覆盖证据。',
            '- 全成员集合被冻结，没有根据测试表现剔除成员。未使用旧原始GLM预测；只读取修复后冻结参数的测试返回包。',
            '- 只完成Shiode R1-5模型部分的重分析。图2正式稿、其他站点固定深度分析及其他审稿意见不由本包自动完成。',
            '- 本脚本不自动写“不影响结论”；是否一致需要依据结果及覆盖范围逐项判断。',
            '', '输入哈希、成员来源、共同掩码、排除窗口及复现脚本都在返回包内。']
    return '\n'.join(lines)+'\n'


def run(args):
    paths=[Path(args.step1a),Path(args.ml_zip),Path(args.glm_tar)]
    for p in paths:require(p.is_file(),f'Missing input: {p}')
    out=Path(args.out) if args.out else BASE/'results'/('shiode_step1B_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    print('Output:',out,flush=True)
    try:
        s,ref,t,raw,q,clean,catalogs,audit_A,month=load_observations(paths[0])
        print('Step1A catalogs loaded; no event re-detection.',flush=True)
        glm,ga=load_glm(paths[2],t,raw)
        ml,ma,dm=load_ml(paths[1],t,raw,s['input_sha256'],args.require_verified_coverage)
        members=glm+ml
        require(len(members)==40,'Expected 40 total members.')
        for model in MODEL_NAMES:require(sum(m.model==model for m in members)==10,'Expected 10 members per model.')
        all_valid=np.logical_and.reduce([m.valid for m in members])
        obs_required_covered=np.all(~q|all_valid,axis=1)
        A_cont=(q.sum(axis=1)>=MIN_LAYERS_CONTINUOUS)&obs_required_covered
        B_cont=q.all(axis=1)&all_valid.all(axis=1)
        print('All 40 existing prediction arrays aligned. Computing diagnostics only.',flush=True)
        ensembles=[]
        for model in MODEL_NAMES:
            group=[m for m in members if m.model==model]
            p=np.mean(np.stack([m.pred for m in group],axis=0),axis=0)
            v=np.logical_and.reduce([m.valid for m in group])
            p[~v]=np.nan
            ensembles.append(Member(model,'temperature_mean_of_fixed10',p,v,dict()))
        elig=[];accepted={}
        for cat,df in catalogs.items():
            kept=[]
            for r in df.itertuples(index=False):
                i,j=int(r.start_index),int(r.end_index)
                required=q[i:j+1]
                absent=[f'{m.model}/{m.member}' for m in members if np.any(required & ~m.valid[i:j+1])]
                ok=not absent
                if cat.startswith('B_'):require(required.all(),'Strict event must use 11 layers throughout.')
                elig.append(dict(catalog=cat,event_id=r.event_id,start_time=r.start_time,end_time=r.end_time,
                                 included_in_all_model_comparison=ok,
                                 reason='included' if ok else 'missing_prediction_on_required_observation_cells',
                                 missing_members=';'.join(absent),required_temperature_cells=int(required.sum()),
                                 minimum_observed_layers=int(required.sum(axis=1).min()),
                                 coverage_scope='includes_CSV_only_members_under_declared_condition'))
                if ok:kept.append(r)
            accepted[cat]=kept
        save_csv(out/'event_eligibility.csv',elig)
        for cat in catalogs:save_csv(out/f'input_events_{cat}.csv',catalogs[cat])
        save_csv(out/'step1A_audit_A_event_windows.csv',audit_A)
        save_csv(out/'step1A_coverage_by_month.csv',month)
        save_csv(out/'prediction_sources.csv',ga+ma)
        save_csv(out/'existing_depth_mapping.csv',dm)
        obs_sd=row_std(clean)
        # This validates stored catalog amplitudes only; detection is not rerun.
        member_event_rows=[];ensemble_event_rows=[];struct_rows=[]
        obs_structure=structure(raw,q)
        obs_g_legacy=legacy_gmean(raw,q,A_cont)
        allowed_by_scope={'A_available_adjacent_no_interp':A_cont,
                          'A_legacy_Gmean_interpolation_comparator':A_cont,
                          'B_fixed11_no_interp':B_cont}
        save_series={'time':t,'obs_STD_available':obs_sd,
                     'obs_STD_fixed11':np.where(q.all(axis=1),obs_sd,np.nan),
                     'obs_Gmean_adjacent':obs_structure[1],
                     'obs_Gmean_legacy_interpolated_comparator':obs_g_legacy}
        for aggregation,group,target in [('per_member',members,member_event_rows),
                                          ('temperature_mean_fixed10',ensembles,ensemble_event_rows)]:
            for m in group:
                sd=row_std(np.where(q,m.pred,np.nan))
                sd[~np.all(~q|m.valid,axis=1)]=np.nan
                for cat,events in accepted.items():
                    for r in events:
                        target.append(event_record(sd,obs_sd,r,m.model,m.member,cat,aggregation,t))
                ps,pg=structure(m.pred,q)
                pg_legacy=legacy_gmean(m.pred,q,A_cont)
                for scope,allowed in allowed_by_scope.items():
                    gs=pg_legacy if scope.startswith('A_legacy') else pg
                    go=obs_g_legacy if scope.startswith('A_legacy') else obs_structure[1]
                    for metric,pred,obs in [('STD',ps,obs_structure[0]),('Gmean',gs,go)]:
                        stats=series_metrics(pred,obs,allowed)
                        struct_rows.append(dict(scope=scope,aggregation=aggregation,model=m.model,
                                                member=m.member,diagnostic=metric,**stats))
                require(np.allclose(pg[B_cont],pg_legacy[B_cont],rtol=0,atol=1e-10),
                        'On complete 11-depth rows, legacy/direct Gmean must agree.')
                if aggregation=='temperature_mean_fixed10':
                    save_series[m.model+'_STD_available']=sd
                    save_series[m.model+'_STD_fixed11']=np.where(B_cont,sd,np.nan)
                    save_series[m.model+'_Gmean_fixed11']=np.where(B_cont,pg,np.nan)
        ef=pd.DataFrame(ensemble_event_rows);mf=pd.DataFrame(member_event_rows);sf=pd.DataFrame(struct_rows)
        require(not ef.empty,'No common events; see event_eligibility.csv. Stop before reporting model results.')
        em=event_summary(ef,'temperature_mean_fixed10');mm=event_summary(mf,'per_member')
        ma_sum=aggregate_member_summaries(mm,['catalog','model'],
                         ['drop_ratio_mean','drop_ratio_median','absolute_ratio_error_mean',
                          'absolute_amplitude_error_degC_mean'],'N_events')
        sa_sum=aggregate_member_summaries(sf[sf.aggregation=='per_member'],['scope','model','diagnostic'],
                                         ['RMSE','MAE','Bias','R','VR'],'N_time')
        save_csv(out/'event_metrics_ensemble_temperature_mean.csv',ef)
        save_csv(out/'event_metrics_each_member.csv',mf)
        save_csv(out/'event_summary_ensemble_temperature_mean.csv',em)
        save_csv(out/'event_summary_each_member.csv',mm)
        save_csv(out/'event_summary_across_members.csv',ma_sum)
        save_csv(out/'continuous_structure_each_member_and_ensemble.csv',sf)
        save_csv(out/'continuous_structure_summary_across_members.csv',sa_sum)
        delta=[]
        for model in MODEL_NAMES:
            a=em[(em.model==model)&(em.catalog=='A_quality_available')]
            b=em[(em.model==model)&(em.catalog=='B_fixed11_complete3h')]
            if len(a)==1 and len(b)==1:
                ar=a.iloc[0];br=b.iloc[0]
                r=dict(model=model,N_A=int(ar.N_events),N_B=int(br.N_events))
                for k in ['drop_ratio_mean','drop_ratio_median','absolute_ratio_error_mean','absolute_amplitude_error_degC_mean']:
                    r[k+'_A']=ar[k];r[k+'_B']=br[k];r[k+'_B_minus_A']=br[k]-ar[k]
                delta.append(r)
        save_csv(out/'A_vs_B_ensemble_event_comparison.csv',delta)
        coverage=[]
        months=pd.DatetimeIndex(t).strftime('%Y-%m')
        for month_name in sorted(set(months)):
            ix=months==month_name
            coverage.append(dict(month=month_name,expected_time_points=int(ix.sum()),
                common_continuous_A_time_points=int(A_cont[ix].sum()),
                common_continuous_B_time_points=int(B_cont[ix].sum()),
                common_A_events=sum(str(r.start_time).startswith(month_name) for r in accepted['A_quality_available']),
                common_B_events=sum(str(r.start_time).startswith(month_name) for r in accepted['B_fixed11_complete3h'])))
        save_csv(out/'common_coverage_by_month.csv',coverage)
        np.savez_compressed(out/'common_scoring_masks.npz',time=t,depths=DEPTHS,
                            observation_valid=q,all_member_prediction_valid=all_valid,
                            continuous_A=A_cont,continuous_B=B_cont,
                            A_event_starts=np.array([r.start_index for r in accepted['A_quality_available']],int),
                            B_event_starts=np.array([r.start_index for r in accepted['B_fixed11_complete3h']],int))
        np.savez_compressed(out/'ensemble_structure_series.npz',**save_series)
        summary=dict(version=VERSION,status='completed_existing_prediction_diagnostics',
                     site='Shiode',time_start=str(t[0]),time_end=str(t[-1]),
                     no_training=True,no_inference=True,no_GLM_run=True,no_event_redetection=True,
                     event_primary_aggregation='temperature_mean_of_all_frozen10_then_STD',
                     event_amplitude_definition='max(STD)-min(STD) inside observed 3h window; model temporal order recorded, not enforced',
                     independent_model_event_occurrence_or_timing_evaluated=False,
                     continuous_A_time_count=int(A_cont.sum()),continuous_B_time_count=int(B_cont.sum()),
                     catalog_A_input_count=len(catalogs['A_quality_available']),catalog_B_input_count=len(catalogs['B_fixed11_complete3h']),
                     catalog_A_common_count=len(accepted['A_quality_available']),catalog_B_common_count=len(accepted['B_fixed11_complete3h']),
                     members={model:[m.member for m in members if m.model==model] for model in MODEL_NAMES},
                     coverage_evidence_incomplete=[m.metadata for m in ml if not m.metadata['independent_counts']],
                     seed_sets_not_changed=True,
                     input_sha256={p.name:sha256(p) for p in paths},
                     limitations=['CSV-only coverage remains conditional; no independent counts invented.',
                                  'A/B evaluate different observation subsets; changes are not new model training gains.',
                                  'Nominal fixed channels do not prove exact physical depth invariance.',
                                  'R1-5 not closed for all sites; figures and wording still need updating.'])
        write_json(out/'summary.json',summary)
        (out/'REPORT_zh.md').write_text(report_text(summary,em,sa_sum),encoding='utf-8')
        (out/'evaluate_shiode_fixed_depth_models.py').write_bytes(Path(__file__).read_bytes())
        # Only generated reports/tables/masks; excludes all large source input archives.
        result_zip=out/'Shiode_R1_5_step1B_return.zip'
        with zipfile.ZipFile(result_zip,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.iterdir()):
                if f.is_file() and f!=result_zip:z.write(f,f.name)
        print('COMPLETE. Read REPORT_zh.md before drawing conclusions.',flush=True)
        print('UPLOAD THIS FILE:',result_zip,flush=True)
    except Exception as exc:
        (out/'FAILED.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('FAILED:',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        print('No input files were modified. No final return ZIP was produced.',file=sys.stderr,flush=True)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--step1a',type=Path,default=BASE/'data/Shiode_R1_5_step1A_return.zip')
    p.add_argument('--ml-zip',type=Path,default=BASE/'data/Shiode_ML_final_members_20261025_181111_032810.zip')
    p.add_argument('--glm-tar',type=Path,default=BASE/'data/GLM_shiode_test_return.tar.gz')
    p.add_argument('--out',type=Path,help='New output directory; must not already exist.')
    p.add_argument('--require-verified-coverage',action='store_true',
                   help='Stop if any member lacks an independent coverage/counts record.')
    run(p.parse_args())

if __name__=='__main__':
    try:main()
    except Exception:sys.exit(1)
