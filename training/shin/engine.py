"""Training/evaluation engine. No test loss is computed during model selection."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from models import Seq2SeqLSTMWithT0, StructureAwareLSTMWithT0, ResidualSSM
from losses import (masked_mse, masked_mean, smoothness_loss_time, smoothness_loss_depth,
                    range_penalty, profile_std_torch, profile_gmean_torch, normalize_by_obs)
from settings import configuration, PAPER_DEPTH_MAX, VERSION
from shin_data import prepare, Windows, file_sha256, target_coverage


def safe_json(value):
    if isinstance(value,dict): return {str(k):safe_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [safe_json(v) for v in value]
    if isinstance(value,np.ndarray): return safe_json(value.tolist())
    if isinstance(value,np.generic): return safe_json(value.item())
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,Path): return str(value)
    return value


def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(safe_json(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def atomic_save(path, value):
    path=Path(path); tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(value,tmp); os.replace(tmp,path)


def implementation_hash():
    root=Path(__file__).resolve().parent
    h=hashlib.sha256()
    for p in sorted(root.glob('*.py')):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(safe_json(obj),sort_keys=True).encode()).hexdigest()


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def configure_device(device='cuda', threads=1, memory_fraction=.28):
    torch.set_num_threads(threads)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Do not silently run the 30 formal fits on CPU.')
        dev=torch.device(device)
        torch.cuda.set_device(dev)
        torch.cuda.set_per_process_memory_fraction(memory_fraction,dev)
        return dev
    return torch.device(device)


def environment():
    return dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__,
                cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                precision='FP32; AMP and TF32 disabled', deterministic=True)


def build_model(model,cfg,n_inputs,n_depths):
    if model=='lstm': return Seq2SeqLSTMWithT0(n_inputs,n_depths,hidden_dim=cfg['hidden'],num_lstm_layers=1,dropout=0.)
    if model=='pgnn': return StructureAwareLSTMWithT0(n_inputs,n_depths,hidden_dim=cfg['hidden'],num_lstm_layers=1,dropout=0.)
    if model=='rssm': return ResidualSSM(n_inputs,n_depths,hidden_dim=cfg['hidden'],update_scale=cfg['update_scale'])
    raise ValueError(model)


def objective(pred,y,mask,depths,cfg,training):
    """Original training objective; validation uses only its own target interval.

    SGH diagnostic normalization remains the source's masked batch-target normalization.
    It never changes model inputs, meteo normalization or T0.
    """
    w=cfg['warmup']; pt=pred[:,w:]; yt=y[:,w:]; mt=mask[:,w:]
    zero=pred.new_zeros(())
    loss_t=masked_mse(pt,yt,mt)
    loss_w=masked_mse(pred[:,:w],y[:,:w],mask[:,:w]) if training else zero
    obs=loss_t+(cfg['warm_weight']*loss_w if training else 0.)
    reg_pred=pred if training else pt
    ts=smoothness_loss_time(reg_pred) if cfg['time_weight'] else zero
    zs=smoothness_loss_depth(reg_pred) if cfg['depth_weight'] else zero
    rg=range_penalty(reg_pred,cfg['temp_min'],cfg['temp_max']) if cfg['range_weight'] else zero
    ls=lg=zero
    if cfg['std_weight'] or cfg['gmean_weight']:
        vt=(mt.sum(dim=-1)>=3).float()
        sp=profile_std_torch(pt,mt); so=profile_std_torch(yt,mt)
        gp=profile_gmean_torch(pt,mt,depths); go=profile_gmean_torch(yt,mt,depths)
        sp,so=normalize_by_obs(sp,so,vt); gp,go=normalize_by_obs(gp,go,vt)
        ls=masked_mean((sp-so)**2,vt); lg=masked_mean((gp-go)**2,vt)
    total=obs+cfg['time_weight']*ts+cfg['depth_weight']*zs+cfg['range_weight']*rg+cfg['std_weight']*ls+cfg['gmean_weight']*lg
    return total,dict(loss=total,obs=obs,warmup_obs=loss_w,target_obs=loss_t,t_smooth=ts,z_smooth=zs,
                      range=rg,structure=ls,dstructure=lg)


def epoch_pass(model,loader,device,cfg,depths,optimizer=None,max_batches=None):
    training=optimizer is not None; model.train(training)
    sums={}; nb=0; se=0.; n=0; skipped=0
    with torch.set_grad_enabled(training):
        for k,(T0,x,y,m,_) in enumerate(loader):
            if max_batches is not None and k>=max_batches: break
            T0,x,y,m=[a.to(device,non_blocking=True) for a in (T0,x,y,m)]
            if not torch.any(m[:,cfg['warmup']:] > .5):
                skipped+=1; continue
            if training: optimizer.zero_grad(set_to_none=True)
            pred=model(T0,x)
            if not torch.isfinite(pred).all(): raise FloatingPointError('Nonfinite model prediction; member not skipped.')
            loss,parts=objective(pred,y,m,depths,cfg,training)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite objective; member not skipped.')
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
                optimizer.step()
            for key,v in parts.items(): sums[key]=sums.get(key,0.)+float(v.detach())
            mt=m[:,cfg['warmup']:]; d=(pred[:,cfg['warmup']:]-y[:,cfg['warmup']:]).detach()
            se+=float(((d*d)*mt).sum()); n+=int(mt.sum()); nb+=1
    if not nb or not n: raise ValueError('No valid objective batches/target observations.')
    return dict({k:v/nb for k,v in sums.items()},rmse=math.sqrt(se/n),n_valid_target_points=n,
                n_batches=nb,skipped_empty_target_batches=skipped)


def cpu_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def rng_state(generator):
    ns=np.random.get_state()
    return dict(python=random.getstate(),numpy_kind=ns[0],numpy_keys=torch.tensor(ns[1].astype(np.int64)),
                numpy_pos=ns[2],numpy_has_gauss=ns[3],numpy_cached_gauss=ns[4],
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loader=generator.get_state())


def restore_rng(rng,generator):
    random.setstate(rng['python'])
    np.random.set_state((rng['numpy_kind'],rng['numpy_keys'].cpu().numpy().astype(np.uint32),rng['numpy_pos'],
                         rng['numpy_has_gauss'],rng['numpy_cached_gauss']))
    torch.set_rng_state(rng['torch'].cpu()); generator.set_state(rng['loader'].cpu())
    if torch.cuda.is_available() and rng['cuda']: torch.cuda.set_rng_state_all([v.cpu() for v in rng['cuda']])


def loaders(data,cfg,seed,device):
    gen=torch.Generator().manual_seed(seed)
    vgen=torch.Generator().manual_seed(0)
    ds_tr=Windows(data,cfg,'train'); ds_va=Windows(data,cfg,'val')
    kwargs=dict(batch_size=cfg['batch_size'],num_workers=0,pin_memory=device.type=='cuda',drop_last=False)
    return (DataLoader(ds_tr,shuffle=True,generator=gen,**kwargs),
            DataLoader(ds_va,shuffle=False,generator=vgen,**kwargs),gen)


def write_prepared(data,cfg,out):
    out.mkdir(parents=True,exist_ok=True)
    r=copy.deepcopy(data.report)
    for split in ('train','val','test'):
        ds=Windows(data,cfg,split); count=target_coverage(ds.starts,len(data.time),cfg)
        r[split]['n_windows']=len(ds)
        ix=np.flatnonzero(count)
        r[split]['first_target']=str(data.time[ix[0]])
        r[split]['last_target']=str(data.time[ix[-1]])
        r[split]['n_target_times']=len(ix)
    r['historical_forcing_warmup_only']=True
    r['formal_test_metrics_during_training']=False
    r['source_hash']=implementation_hash()
    atomic_json(out/'data_split_report.json',r)
    np.savez_compressed(out/'normalization_and_T0.npz',mean=data.meteo_mean,std=data.meteo_std,T0=data.T0,
                        depths=data.depths,train_end=data.train_end,dev_end=data.dev_end)
    atomic_json(out/'environment.json',environment())
    return r


def train_one(data,cfg,member,seed,out,device,resume=False,stop_after=None):
    """stop_after is for unit tests only; it never changes the planned number of epochs."""
    out=Path(out)
    record=dict(config=cfg,member_id=member,actual_seed=seed,data_sha256=data.source_sha256,
                torch_version=str(torch.__version__),numpy_version=np.__version__,pandas_version=pd.__version__,
                implementation_sha256=implementation_hash())
    ident=fingerprint(record)
    if out.exists():
        config_path=out/'config.json'
        if not resume:
            raise FileExistsError(f'{out} exists. Use explicit --resume; no overwriting old experiments.')
        if not config_path.is_file() or json.loads(config_path.read_text())['fingerprint']!=ident:
            raise ValueError(f'Resume configuration/source/input mismatch at {out}')
        if (out/'TRAINING_DONE.json').is_file():
            print(f'[SKIP completed] {cfg["model"]} member={member} actual_seed={seed}',flush=True)
            return json.loads((out/'TRAINING_DONE.json').read_text())
    else:
        out.mkdir(parents=True)
        atomic_json(out/'config.json',dict(record,fingerprint=ident))
    seed_everything(seed)
    model=build_model(cfg['model'],cfg,data.x.shape[1],len(data.depths)).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    tr,va,gen=loaders(data,cfg,seed,device)
    depths=torch.as_tensor(data.depths,dtype=torch.float32,device=device)
    history={'train':[],'val':[]}
    best=float('inf'); best_train=float('inf'); best_epoch=0; best_train_epoch=0
    best_state=None; best_train_state=None; start=1
    ckpt_path=out/'last_checkpoint.pt'
    if resume and ckpt_path.is_file():
        ck=torch.load(ckpt_path,map_location='cpu',weights_only=True)
        if ck['fingerprint']!=ident: raise ValueError('Checkpoint fingerprint mismatch.')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        start=ck['epoch']+1; best=ck['best_val']; best_epoch=ck['best_epoch']
        best_train=ck['best_train']; best_train_epoch=ck['best_train_epoch']
        best_state=ck['best_model']; best_train_state=ck['best_train_model']; history=ck['history']
        atomic_save(out/'best_model.pt',best_state)
        atomic_save(out/'best_train_model.pt',best_train_state)
        restore_rng(ck['rng'],gen)
        print(f'[RESUME] {cfg["model"]} seed={seed} epoch={start}',flush=True)
    tic=time.perf_counter()
    print(f'[START] model={cfg["model"]} member={member} actual_seed={seed} params={sum(p.numel() for p in model.parameters())} train_windows={len(tr.dataset)} val_windows={len(va.dataset)}',flush=True)
    for epoch in range(start,cfg['epochs']+1):
        e0=time.perf_counter()
        train=epoch_pass(model,tr,device,cfg,depths,opt)
        val=epoch_pass(model,va,device,cfg,depths)
        history['train'].append(dict(epoch=epoch,**train))
        history['val'].append(dict(epoch=epoch,**val))
        if val['loss']<best:
            best=val['loss']; best_epoch=epoch; best_state=cpu_state(model)
            atomic_save(out/'best_model.pt',best_state)
        if train['loss']<best_train:
            best_train=train['loss']; best_train_epoch=epoch; best_train_state=cpu_state(model)
            atomic_save(out/'best_train_model.pt',best_train_state)
        current=cpu_state(model)
        atomic_save(out/'last_model.pt',current)
        if cfg.get('checkpoint_every',0) and epoch%cfg['checkpoint_every']==0:
            atomic_save(out/f'epoch_{epoch:03d}.pt',current)
        checkpoint=dict(epoch=epoch,model=current,optimizer=opt.state_dict(),rng=rng_state(gen),history=history,
                        best_val=best,best_epoch=best_epoch,best_train=best_train,best_train_epoch=best_train_epoch,
                        best_model=best_state,best_train_model=best_train_state,fingerprint=ident)
        atomic_save(ckpt_path,checkpoint)
        atomic_json(out/'history.json',history)
        atomic_json(out/'status.json',dict(epoch=epoch,planned_epochs=cfg['epochs'],best_epoch=best_epoch,
                                          best_val_loss=best,seconds_last_epoch=time.perf_counter()-e0,
                                          elapsed_session_seconds=time.perf_counter()-tic))
        if epoch==1 or epoch%10==0 or epoch==cfg['epochs']:
            print(f'[{cfg["model"]} seed={seed} {epoch:03d}/{cfg["epochs"]}] train={train["loss"]:.6f} val={val["loss"]:.6f} val_RMSE={val["rmse"]:.6f} best_epoch={best_epoch} sec={time.perf_counter()-e0:.1f}',flush=True)
        if stop_after is not None and epoch>=stop_after:
            return {'interrupted_for_test':True,'epoch':epoch}
    if best_state is None: raise RuntimeError('No best validation checkpoint.')
    done=dict(model=cfg['model'],member_id=member,actual_seed=seed,best_epoch=best_epoch,best_val_loss=best,
              best_train_epoch=best_train_epoch,completed_epochs=cfg['epochs'],fingerprint=ident,
              selection='minimum internal validation target-only composite loss; never test',
              selected_checkpoint_sha256=file_sha256(out/'best_model.pt'))
    atomic_json(out/'TRAINING_DONE.json',done)
    return done


def metrics(pred,obs,mask):
    good=np.asarray(mask,dtype=bool)&np.isfinite(pred)&np.isfinite(obs)
    p=np.asarray(pred,dtype=np.float64)[good]; o=np.asarray(obs,dtype=np.float64)[good]
    if not len(p): return dict(N=0,RMSE=None,R=None,NSE=None,Bias=None)
    se=np.sum((p-o)**2); denom=np.sum((o-o.mean())**2)
    r=float(np.corrcoef(p,o)[0,1]) if len(o)>1 and np.std(o)>0 and np.std(p)>0 else np.nan
    return safe_json(dict(N=len(p),RMSE=float(np.sqrt(se/len(p))),R=r,
                          NSE=float(1-se/denom) if denom>0 else np.nan,Bias=float(np.mean(p-o))))


@torch.no_grad()
def predict_segment(model,data,cfg,device,split='test'):
    model.eval()
    ds=Windows(data,cfg,split)
    dl=DataLoader(ds,batch_size=cfg['batch_size'],shuffle=False,num_workers=0,pin_memory=device.type=='cuda')
    sums=np.zeros((len(data.time),len(data.depths)),dtype=np.float64)
    count=np.zeros(len(data.time),dtype=np.int32)
    for T0,x,_,_,starts in dl:
        p=model(T0.to(device),x.to(device))[:,cfg['warmup']:].cpu().numpy()
        if not np.isfinite(p).all(): raise FloatingPointError('Nonfinite held-out prediction.')
        for j,s in enumerate(starts.tolist()):
            a=s+cfg['warmup']; b=a+cfg['target']; sums[a:b]+=p[j]; count[a:b]+=1
    out=np.full(sums.shape,np.nan,dtype=np.float32)
    covered=count>0; out[covered]=(sums[covered]/count[covered,None]).astype(np.float32)
    lo,hi=(data.train_end,data.dev_end) if split=='val' else (data.dev_end,len(data.time))
    if not np.all(covered[lo:hi]): raise RuntimeError('Uncovered target tail; output cannot be treated as zeros.')
    return out[lo:hi],count[lo:hi]


def evaluate_all(data,cfg,out,device):
    """Must run only after all fixed members of this model have finished training."""
    out=Path(out); rows=[]; rows_all=[]; layers=[]
    frozen=[]
    for member,seed in enumerate(cfg['actual_seeds']):
        d=out/f'member_{member:02d}_seed{seed}'
        if not (d/'TRAINING_DONE.json').is_file(): raise RuntimeError(f'Missing completed member: {d}')
        done=json.loads((d/'TRAINING_DONE.json').read_text())
        if file_sha256(d/'best_model.pt')!=done['selected_checkpoint_sha256']:
            raise RuntimeError(f'Best checkpoint changed after freezing: {d}')
        frozen.append(done)
    atomic_json(out/'frozen_members.json',frozen)
    exports=out/'exports'/cfg['export_folder']; exports.mkdir(parents=True,exist_ok=True)
    paper=data.depths<=PAPER_DEPTH_MAX+1e-6
    obs=data.temp[data.dev_end:].copy(); valid=data.valid[data.dev_end:]
    obs[~valid]=np.nan
    for member,seed in enumerate(cfg['actual_seeds']):
        d=out/f'member_{member:02d}_seed{seed}'; done=frozen[member]
        model=build_model(cfg['model'],cfg,data.x.shape[1],len(data.depths)).to(device)
        model.load_state_dict(torch.load(d/'best_model.pt',map_location=device,weights_only=True))
        pred,count=predict_segment(model,data,cfg,device,'test')
        covered=np.broadcast_to(count[:,None]>0,pred.shape)
        mask=valid&covered&np.isfinite(pred)
        np.savez_compressed(d/'test_predictions.npz',time=data.time[data.dev_end:],depths=data.depths,
                            pred=pred,obs=obs,obs_valid=valid,prediction_coverage=covered,
                            evaluation_mask=mask,count=count,T0=data.T0,actual_seed=seed,member_id=member)
        # Keep actual RNG seed distinct from historical plotting member ID ("seed0" was often RNG 42).
        tab=pd.DataFrame({'time':pd.to_datetime(data.time[data.dev_end:])})
        for j,z in enumerate(data.depths): tab[f'{z:.2f}m_pred']=pred[:,j]
        tab.to_csv(exports/f'shin_{cfg["model"]}_member{member:02d}_seed{seed}.csv',index=False)
        q=metrics(pred[:,paper],obs[:,paper],mask[:,paper])
        row=dict(model=cfg['model'],member_id=member,actual_seed=seed,best_epoch=done['best_epoch'],**q)
        rows.append(row); rows_all.append(dict(model=cfg['model'],member_id=member,actual_seed=seed,
                                             **metrics(pred,obs,mask)))
        for j,z in enumerate(data.depths):
            layers.append(dict(model=cfg['model'],member_id=member,actual_seed=seed,depth_m=z,
                               **metrics(pred[:,j],obs[:,j],mask[:,j])))
        atomic_json(d/'test_metrics.json',dict(paper_17_channels=row,all_18_channels=rows_all[-1],
                                              common_GLM_mask_applied=False))
        print(f'[TEST] {cfg["model"]} member={member} actual_seed={seed} RMSE={q["RMSE"]:.6f}',flush=True)
        del model
        if device.type=='cuda': torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out/'test_metrics_by_member.csv',index=False)
    pd.DataFrame(rows_all).to_csv(out/'test_metrics_all18_by_member.csv',index=False)
    pd.DataFrame(layers).to_csv(out/'test_metrics_by_layer.csv',index=False)
    agg={}
    for k in ('RMSE','R','NSE','Bias'):
        v=np.array([r[k] for r in rows],dtype=float)
        agg[k]={'mean':float(v.mean()),'sample_sd':float(v.std(ddof=1))}
    atomic_json(out/'test_summary.json',dict(model=cfg['model'],n_members=len(rows),members=rows,
                  aggregate=agg,aggregation='mean and sample SD of individual-member metrics',
                  scope='17 nominal channels 0.25..3.45m; ML data-only QC; NOT yet GLM common mask',
                  n_time=len(data.time)-data.dev_end,test_start=str(data.time[data.dev_end]),
                  test_end=str(data.time[-1]),all_prediction_targets_covered=True))
    atomic_json(out/'EVALUATION_DONE.json',{'version':VERSION,'n_members':len(rows)})


def cli(model):
    ap=argparse.ArgumentParser(description=f'Shin {model}: 56/14/30 chronological retraining')
    ap.add_argument('--csv',type=Path,default=Path('data/shin_2025_aligned_qc_wide.csv'))
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--phase',choices=['preflight','smoke','train','evaluate'],default='train')
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--memory-fraction',type=float,default=.28)
    ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args(); cfg=configuration(model)
    dev=configure_device(args.device,args.threads,args.memory_fraction)
    data=prepare(args.csv)
    if args.phase=='preflight':
        r=write_prepared(data,cfg,args.out)
        print(json.dumps(r,ensure_ascii=False,indent=2)); return
    if args.phase=='smoke':
        seed_everything(cfg['actual_seeds'][0]); m=build_model(model,cfg,data.x.shape[1],len(data.depths)).to(dev)
        opt=torch.optim.Adam(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        tr,va,_=loaders(data,cfg,cfg['actual_seeds'][0],dev)
        z=torch.as_tensor(data.depths,dtype=torch.float32,device=dev)
        before=cpu_state(m)
        train=epoch_pass(m,tr,dev,cfg,z,opt,max_batches=1); val=epoch_pass(m,va,dev,cfg,z,max_batches=1)
        if not any(not torch.equal(before[k],v.cpu()) for k,v in m.state_dict().items()):
            raise AssertionError('Smoke optimizer did not change model parameters.')
        args.out.mkdir(parents=True,exist_ok=True)
        saved=args.out/'smoke_weights.pt'; atomic_save(saved,cpu_state(m))
        clone=build_model(model,cfg,data.x.shape[1],len(data.depths)).to(dev)
        clone.load_state_dict(torch.load(saved,map_location=dev,weights_only=True))
        report=dict(model=model,train=train,val=val,configured_batch_size=cfg['batch_size'],actual_train_batch_size=min(cfg['batch_size'],len(tr.dataset)),full_length=cfg['warmup']+cfg['target'],
                    max_cuda_allocated_GiB=torch.cuda.max_memory_allocated(dev)/2**30 if dev.type=='cuda' else None,
                    max_cuda_reserved_GiB=torch.cuda.max_memory_reserved(dev)/2**30 if dev.type=='cuda' else None,
                    forward_backward_adam_save_load_passed=True)
        atomic_json(args.out/'smoke_pass.json',report); print(json.dumps(report),flush=True); return
    if args.phase=='evaluate':
        evaluate_all(data,cfg,args.out,dev); return
    if (args.out/'data_split_report.json').exists() and not args.resume:
        raise FileExistsError(f'Output exists: {args.out}; use --resume or a new path.')
    write_prepared(data,cfg,args.out)
    for member,seed in enumerate(cfg['actual_seeds']):
        train_one(data,cfg,member,seed,args.out/f'member_{member:02d}_seed{seed}',dev,args.resume)
        if dev.type=='cuda': torch.cuda.empty_cache()
    atomic_json(args.out/'ALL_TRAINING_DONE.json',dict(model=model,actual_seeds=cfg['actual_seeds'],
                                                   planned_epochs=cfg['epochs']))

if __name__=='__main__':
    raise SystemExit('Run train_lstm.py, train_rssm.py or train_pgnn.py instead.')
