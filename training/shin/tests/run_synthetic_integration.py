import os,sys,json,subprocess,time
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import torch,pandas as pd
from test_protocol import frame,tiny
from shin_data import prepare_frame
from engine import train_one,evaluate_all,write_prepared,configure_device,atomic_json
import tempfile
_tmp=tempfile.TemporaryDirectory(prefix='shin_synthetic_test_')
out=Path(_tmp.name)
a=prepare_frame(frame(n=180));device=configure_device('cpu',1);tic=time.perf_counter()
for model in ('lstm','rssm','pgnn'):
 c=tiny(model);c['epochs']=1;d=out/'models'/model;write_prepared(a,c,d)
 for member,seed in enumerate(c['actual_seeds']):train_one(a,c,member,seed,d/f'member_{member:02d}_seed{seed}',device)
 atomic_json(d/'ALL_TRAINING_DONE.json',{'model':model})
# All 30 checkpoint files fixed before any held-out metrics exist.
assert len(list(out.glob('models/*/member_*/TRAINING_DONE.json')))==30
assert not list(out.glob('models/*/member_*/test_predictions.npz'))
atomic_json(out/'ALL_30_MODELS_FROZEN.json',{'synthetic_integration_test':True})
for model in ('lstm','rssm','pgnn'):
 c=tiny(model);c['epochs']=1;evaluate_all(a,c,out/'models'/model,device)
subprocess.run([sys.executable,str(ROOT/'collect_results.py'),'--run',str(out)],check=True)
t=pd.read_csv(out/'all_models_test_metrics_by_member.csv')
assert len(t)==30
assert all(t.groupby('model').size()==10)
assert t['N'].nunique()==1
assert len(list(out.glob('models/*/exports/*/*.csv')))==30
assert len(list(out.glob('models/*/member_*/test_predictions.npz')))==30
assert (out/'Shin_ML_retrain_return.tar.gz').is_file()
print(json.dumps({'integration':'30 synthetic fits + freeze barrier + 30 evaluations + result archive','all_pass':True,'seconds':time.perf_counter()-tic}))
