"""Small function tests only. No user observations/predictions or model execution."""
import io
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
import evaluate_shiode_fixed_depth_models as m

class FunctionTests(unittest.TestCase):
    def test_align_preserves_missing_time(self):
        t=pd.date_range('2000-01-01',periods=3,freq='10min').to_numpy()
        a,p=m.align(t[[0,2]],np.array([[1.],[3.]]),t)
        self.assertTrue(np.isnan(a[1,0]));self.assertEqual(p.tolist(),[True,False,True])
    def test_duplicate_timestamps_rejected(self):
        with self.assertRaises(ValueError):m.times(['2000-01-01','2000-01-01'])
    def test_exact_depth_no_automatic_shift(self):
        with self.assertRaises(ValueError):m.depth_indices(m.DEPTHS-.05)
    def test_complete_gradient_linear_profile(self):
        x=np.tile(m.DEPTHS*2+10,(3,1));q=np.ones_like(x,bool)
        sd,g=m.structure(x,q,True)
        np.testing.assert_allclose(g,2)
        np.testing.assert_allclose(sd,np.std(x,axis=1))
    def test_no_bridging_for_adjacent_gradient(self):
        x=np.tile(m.DEPTHS*2+10,(2,1));q=np.ones_like(x,bool);q[0,5]=False
        x[0,5]=1000
        sd,g=m.structure(x,q)
        self.assertAlmostEqual(g[0],2)
        self.assertTrue(np.isfinite(sd[0]))
        sd2,_=m.structure(x,q,True);self.assertTrue(np.isnan(sd2[0]))
    def test_minimum_eight_layers(self):
        x=np.tile(m.DEPTHS,(1,1));q=np.ones_like(x,bool);q[:,:4]=False
        sd,g=m.structure(x,q)
        self.assertTrue(np.isnan(sd[0]));self.assertTrue(np.isnan(g[0]))
    def test_legacy_equals_direct_when_complete(self):
        x=np.array([m.DEPTHS**2, m.DEPTHS[::-1]**2])
        q=np.ones_like(x,bool);_,g=m.structure(x,q)
        np.testing.assert_allclose(g,m.legacy_gmean(x,q,np.ones(2,bool)))
    def test_metrics_mask(self):
        p=np.array([2.,4.,np.nan]);o=np.array([1.,3.,np.nan]);mask=np.array([1,1,0],bool)
        d=m.series_metrics(p,o,mask)
        self.assertEqual(d['N_time'],2);self.assertAlmostEqual(d['RMSE'],1.)
    def test_event_range_does_not_fabricate_downward_order(self):
        t=pd.date_range('2000-01-01',periods=19,freq='10min').to_numpy()
        obs=np.linspace(1.,.4,19);pred=np.linspace(.1,.4,19)
        e=SimpleNamespace(start_index=0,end_index=18,peak_index=0,min_index=18,drop_degC=.6,event_id='B1')
        row=m.event_record(pred,obs,e,'test','member','B','temperature_mean_fixed10',t)
        self.assertAlmostEqual(row['drop_ratio'],.5)
        self.assertFalse(row['model_min_after_max'])
        self.assertAlmostEqual(row['signed_change_at_observed_extrema_ratio'],-.5)
    def test_no_pickle(self):
        b=io.BytesIO();np.savez(b,a=np.array([{'unsafe':1}],dtype=object))
        with self.assertRaises(ValueError):m.read_npz(b.getvalue())

if __name__=='__main__':unittest.main(verbosity=2)
