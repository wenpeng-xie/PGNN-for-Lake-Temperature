from __future__ import annotations
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from settings import configuration,METEO_COLS,QC_FLAGS
from shin_data import prepare_frame,split_bounds,Windows,target_coverage
from engine import (build_model,objective,train_one,predict_segment,metrics,configure_device,cpu_state,
                    rng_state,restore_rng,seed_everything)


def frame(n=800, channels=18):
    rng=np.random.default_rng(2026)
    idx=np.arange(n)
    d={'time':pd.date_range('2025-03-01',periods=n,freq='10min')}
    for j,c in enumerate(METEO_COLS): d[c]=10+j+np.sin(idx/35)+rng.normal(0,.05,n)
    depths=[.05]+[round(.25+.2*i,2) for i in range(channels)]
    for j,z in enumerate(depths):
        c=f'{z:.2f}m'; d[c]=20-.1*j+np.sin(idx/27)*(1-.025*j)
        for flag in QC_FLAGS: d[f'{c}_{flag}']=np.zeros(n,dtype=bool)
        d[f'{c}_flag_unstable']=np.ones(n,dtype=bool)
        d[f'{c}_flag_vert']=np.ones(n,dtype=bool)
        d[f'{c}_qc_bad']=np.ones(n,dtype=bool)
    return pd.DataFrame(d)


def tiny(model):
    c=configuration(model);c.update(warmup=8,target=4,stride=3,hidden=4,batch_size=16,epochs=2,checkpoint_every=1)
    return c


def source_definitions(filename,names):
    text=(ROOT/'provenance'/filename).read_text();tree=ast.parse(text)
    selected=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    ns=dict(torch=torch,nn=torch.nn,np=np)
    exec(compile(ast.Module(body=selected,type_ignores=[]),filename,'exec'),ns)
    return ns


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_device('cpu',1)

    def test_01_exact_split(self):
        tr,de=split_bounds(34474)
        self.assertEqual((tr,de),(19304,24131))
        self.assertEqual((tr,de-tr,34474-de),(19304,4827,10343))

    def test_02_original_seeds(self):
        self.assertEqual(configuration('lstm')['actual_seeds'],[42,1,2,3,4,5,6,7,8,9])
        self.assertEqual(configuration('rssm')['actual_seeds'],[42,1,2,3,4,5,6,7,8,9])
        self.assertEqual(configuration('pgnn')['actual_seeds'],list(range(10)))

    def test_03_training_statistics_invariant_to_val_test(self):
        f=frame();a=prepare_frame(f)
        # Change all future weather and target values: training x, norm, T0 and masks stay identical.
        g=f.copy();g.loc[a.train_end:,METEO_COLS]=g.loc[a.train_end:,METEO_COLS]*99+100
        for col in a.columns:g.loc[a.train_end:,col]+=1000
        b=prepare_frame(g)
        for x,y in [(a.x[:a.train_end],b.x[:b.train_end]),(a.T0,b.T0),(a.meteo_mean,b.meteo_mean),
                    (a.meteo_std,b.meteo_std)]: np.testing.assert_array_equal(x,y)

    def test_04_test_changes_do_not_affect_validation(self):
        f=frame();a=prepare_frame(f);g=f.copy()
        g.loc[a.dev_end:,METEO_COLS]=np.nan
        # Give test its own observations to permit independent interpolation.
        g.loc[len(g)-1,METEO_COLS]=1000
        for col in a.columns:g.loc[a.dev_end:,col]=9999
        b=prepare_frame(g)
        np.testing.assert_array_equal(a.x[:a.dev_end],b.x[:b.dev_end])
        np.testing.assert_array_equal(a.temp[:a.dev_end],b.temp[:b.dev_end])

    def test_05_qc_keeps_physical_flags_but_rejects_data_flags(self):
        f=frame();f['0.25m_flag_spike']=f['0.25m_flag_spike'].astype(object);f['0.25m_flag_nan']=f['0.25m_flag_nan'].astype(object);f.loc[1,'0.25m_flag_spike']='True';f.loc[2,'0.25m_flag_nan']='False'
        f.loc[2,'0.25m']=np.nan
        a=prepare_frame(f)
        self.assertTrue(a.valid[0,0]);self.assertFalse(a.valid[1,0]);self.assertFalse(a.valid[2,0])
        self.assertEqual(a.temp[2,0],0)

    def test_06_reject_invalid_qc_encoding(self):
        f=frame();f['0.25m_flag_nan']=f['0.25m_flag_nan'].astype(object);f.loc[0,'0.25m_flag_nan']='maybe'
        with self.assertRaises(ValueError):prepare_frame(f)

    def test_07_train_and_val_windows_do_not_cross_target_boundaries(self):
        a=prepare_frame(frame());c=tiny('lstm')
        tr=Windows(a,c,'train');va=Windows(a,c,'val');te=Windows(a,c,'test')
        self.assertTrue(np.all(tr.starts+c['warmup']+c['target']<=a.train_end))
        self.assertTrue(np.all(va.starts+c['warmup']>=a.train_end))
        self.assertTrue(np.all(va.starts+c['warmup']+c['target']<=a.dev_end))
        self.assertTrue(np.all(te.starts+c['warmup']>=a.dev_end))
        self.assertTrue(np.all(te.starts+c['warmup']+c['target']<=len(a.time)))

    def test_08_validation_warmup_has_no_temperature_labels(self):
        a=prepare_frame(frame());c=tiny('pgnn');ds=Windows(a,c,'val')
        T0,x,y,m,s=ds[0]
        self.assertEqual(int(m[:c['warmup']].sum()),0)
        self.assertEqual(float(y[:c['warmup']].sum()),0)
        np.testing.assert_array_equal(T0,a.T0)
        np.testing.assert_array_equal(x[:c['warmup']],a.x[s:s+c['warmup']])

    def test_09_target_tail_coverage_complete(self):
        a=prepare_frame(frame());c=tiny('rssm')
        for split,lo,hi in [('val',a.train_end,a.dev_end),('test',a.dev_end,len(a.time))]:
            ds=Windows(a,c,split);counts=target_coverage(ds.starts,len(a.time),c)
            self.assertTrue(np.all(counts[lo:hi]>0));self.assertFalse(counts[:lo].any());self.assertFalse(counts[hi:].any())

    def test_10_temperature_channel_count_preserved(self):
        a=prepare_frame(frame());self.assertEqual(len(a.columns),18);self.assertEqual(a.depths[-1],3.65)
        self.assertEqual(np.sum(a.depths<=3.45+1e-6),17)
        with self.assertRaises(ValueError):prepare_frame(frame(channels=17))

    def test_11_reject_gaps_duplicate_or_out_of_order_time(self):
        f=frame()
        for g in [f.drop(index=10),pd.concat([f.iloc[:10],f.iloc[9:]]),f.iloc[::-1]]:
            with self.assertRaises(ValueError):prepare_frame(g)

    def test_12_validation_objective_ignores_all_warmup_values(self):
        torch.manual_seed(2)
        for model in ('lstm','rssm','pgnn'):
            c=tiny(model);p=torch.randn(2,12,4)+20;y=torch.randn_like(p)+20;m=torch.ones_like(p);z=torch.arange(4)*.2+.25
            a,_=objective(p,y,m,z,c,False)
            p2=p.clone();y2=y.clone();p2[:,:8]=20000;y2[:,:8]=-30000
            b,_=objective(p2,y2,m,z,c,False)
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_13_all_model_outputs_match_original_source(self):
        for name,fn,defs,cl in [('lstm','lstm_train.py',{'Seq2SeqLSTMWithT0'},'Seq2SeqLSTMWithT0'),
             ('pgnn','pgnn_train.py',{'StructureAwareLSTMWithT0'},'StructureAwareLSTMWithT0'),
             ('rssm','models.py',{'DeltaTUpdate','ResidualSSM'},'ResidualSSM')]:
            ns=source_definitions(fn,defs);c=configuration(name)
            kwargs=dict(input_dim=8,n_layers=18,hidden_dim=64)
            if name=='rssm':kwargs['update_scale']=.05
            orig=ns[cl](**kwargs);new=build_model(name,c,8,18);new.load_state_dict(orig.state_dict())
            x=torch.randn(2,16,8);T0=torch.rand(2,18)+20
            torch.testing.assert_close(orig(T0,x),new(T0,x),rtol=0,atol=0)
            self.assertEqual(sum(p.numel() for p in new.parameters()),9956 if name=='rssm' else 24722)

    def test_14_training_loss_matches_original_formula(self):
        names={'masked_mean','masked_mse','smoothness_loss_time','smoothness_loss_depth','range_penalty',
               'torch_water_density','profile_std_torch','profile_gmean_torch','profile_heat_content_torch',
               'normalize_by_obs','structure_breakdown_loss'}
        ns=source_definitions('pgnn_train.py',names)
        for model in ('lstm','rssm','pgnn'):
            c=tiny(model);p=(torch.randn(2,12,4)+20).requires_grad_();y=torch.randn_like(p)+20;m=torch.ones_like(p);z=torch.arange(4)*.2+.25
            a,_=objective(p,y,m,z,c,True)
            old=c['warm_weight']*ns['masked_mse'](p[:,:8],y[:,:8],m[:,:8])+ns['masked_mse'](p[:,8:],y[:,8:],m[:,8:])
            old=old+c['time_weight']*ns['smoothness_loss_time'](p)+c['depth_weight']*ns['smoothness_loss_depth'](p)+c['range_weight']*ns['range_penalty'](p)
            if model=='pgnn':
                ss,gg,_,_=ns['structure_breakdown_loss'](p,y,m,z,8);old=old+.30*ss+.20*gg
            torch.testing.assert_close(a,old,rtol=1e-6,atol=1e-6)
            ga=torch.autograd.grad(a,p,retain_graph=True)[0];gb=torch.autograd.grad(old,p)[0]
            torch.testing.assert_close(ga,gb,rtol=1e-5,atol=1e-6)

    def test_15_train_best_save_load_and_evaluate_three_models(self):
        a=prepare_frame(frame(n=180))
        with tempfile.TemporaryDirectory() as td:
            for model in ('lstm','rssm','pgnn'):
                c=tiny(model);p=Path(td)/model
                done=train_one(a,c,0,42,p,torch.device('cpu'))
                h=json.loads((p/'history.json').read_text())
                self.assertEqual(done['best_epoch'],min(h['val'],key=lambda r:r['loss'])['epoch'])
                for fn in ('best_model.pt','best_train_model.pt','last_model.pt','last_checkpoint.pt','epoch_002.pt'):
                    self.assertTrue((p/fn).is_file())
                net=build_model(model,c,8,18);net.load_state_dict(torch.load(p/'best_model.pt',weights_only=True))
                pred,count=predict_segment(net,a,c,torch.device('cpu'))
                self.assertTrue(np.isfinite(pred).all());self.assertTrue(np.all(count>0));self.assertEqual(len(pred),len(a.time)-a.dev_end)
                self.assertFalse((p/'test_predictions.npz').exists()) # train_one never evaluates test

    def test_16_resume_matches_uninterrupted(self):
        a=prepare_frame(frame(n=180));c=tiny('lstm')
        with tempfile.TemporaryDirectory() as td:
            full=Path(td)/'full';res=Path(td)/'res'
            train_one(a,c,0,42,full,torch.device('cpu'))
            train_one(a,c,0,42,res,torch.device('cpu'),stop_after=1)
            train_one(a,c,0,42,res,torch.device('cpu'),resume=True)
            x=torch.load(full/'last_model.pt',weights_only=True);y=torch.load(res/'last_model.pt',weights_only=True)
            for key in x:torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
            self.assertEqual(json.loads((full/'history.json').read_text()),json.loads((res/'history.json').read_text()))

    def test_17_existing_output_refused(self):
        a=prepare_frame(frame(n=180));c=tiny('rssm')
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileExistsError):train_one(a,c,0,42,Path(td),torch.device('cpu'))

    def test_18_resume_data_mismatch_rejected(self):
        a=prepare_frame(frame(n=180));c=tiny('lstm')
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'fit';train_one(a,c,0,42,p,torch.device('cpu'),stop_after=1)
            a.source_sha256='changed'
            with self.assertRaises(ValueError):train_one(a,c,0,42,p,torch.device('cpu'),resume=True)

    def test_19_metric_masks_not_zero_rule(self):
        q=metrics(np.array([0.,999.,2.]),np.array([0.,1.,2.]),np.array([True,False,True]))
        self.assertEqual(q['N'],2);self.assertEqual(q['RMSE'],0.)

    def test_20_no_temperature_filling(self):
        f=frame();f.loc[8:12,'0.25m']=np.nan
        a=prepare_frame(f)
        self.assertFalse(a.valid[8:13,0].any());self.assertTrue(np.all(a.temp[8:13,0]==0))
        # Zero is internal placeholder only, never an imputed temperature target.

if __name__=='__main__': unittest.main(verbosity=2)
