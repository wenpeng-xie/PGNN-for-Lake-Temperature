"""Small synthetic-only tests. Does not read any real lake observations."""
import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import detect_fixed_depth_events as m


def fake_data(site='shin', n=90):
    depths=m.EXPECTED_DEPTHS[site].copy()
    nlayer=len(depths)
    # One artificial cooling transition with no physical interpretation.
    amplitude=np.ones(n)*0.8
    amplitude[10:20]=np.linspace(0.8,0.1,10)
    amplitude[20:]=0.1
    shape=np.linspace(-1,1,nlayer)
    shape=shape/np.std(shape)
    raw=20+amplitude[:,None]*shape[None,:]
    time=pd.date_range('2000-01-01', periods=n, freq='10min')
    return dict(site=site, time=time, depths=depths, raw=raw,
                valid=np.ones(raw.shape,bool), present=np.ones(n,bool),
                unknown=np.zeros(raw.shape,bool), bad=np.zeros(raw.shape,bool),
                source_index=np.arange(n), meta={})


def zip_input(path, bad_mask=False):
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('batch_summary.json',json.dumps(dict(status='completed',n_errors=0)))
        for site in ('shin','yamanokami'):
            d=fake_data(site)
            # Simulate a permanently absent shallow channel in the latter half.
            if site=='yamanokami':
                d['raw'][45:,0]=np.nan
                d['valid'][45:,0]=False
            if bad_mask and site=='shin':
                d['valid'][0,0]=False
            b=io.BytesIO()
            np.savez_compressed(b, time_ns=d['time'].asi8, depths=d['depths'],
                                temperature_raw=d['raw'], valid_data_only=d['valid'],
                                source_row_present=d['present'], required_qc_unknown=d['unknown'],
                                required_qc_bad=d['bad'], joint_time_mask=np.ones(len(d['time']),bool))
            z.writestr(site+'/observations_for_followup.npz',b.getvalue())
            meta=dict(scopes={'joint_evaluation_span':dict(start=str(d['time'][0]),
                 end=str(d['time'][-1]),n_complete_data_quality_profiles=int(d['valid'].all(1).sum()))})
            z.writestr(site+'/summary.json',json.dumps(meta))


class SyntheticTests(unittest.TestCase):
    def test_population_sd(self):
        np.testing.assert_allclose(m.row_std(np.array([[1.,3.]])),[1.])

    def test_missing_sd(self):
        out=m.row_std(np.array([[1,np.nan,3],[np.nan,np.nan,np.nan]]))
        self.assertEqual(out[0],1);self.assertTrue(np.isnan(out[1]))

    def test_19_samples(self):
        self.assertFalse(m.forward_complete(np.ones(18,bool)).any())
        out=m.forward_complete(np.ones(19,bool))
        self.assertTrue(out[0]);self.assertEqual(out.sum(),1)

    def test_interior_gap(self):
        complete=np.ones(40,bool);complete[9]=False
        mask=m.forward_complete(complete)
        self.assertFalse(mask[0]);self.assertTrue(mask[10])

    def test_full_duration_tail(self):
        out=m.forward_complete(np.ones(40,bool))
        self.assertEqual(out.sum(),22);self.assertFalse(out[-18:].any())

    def test_decline_not_recovery(self):
        self.assertTrue(m.candidates(np.linspace(0.8,0.1,19))[0])
        self.assertFalse(m.candidates(np.linspace(0.6,1.2,19))[0])

    def test_gate_before_greedy_merge(self):
        flags=np.zeros(50,bool);flags[[0,18,19,37]]=True
        self.assertEqual(m.merge_starts(flags),[0,19])
        flags[0]=False
        self.assertEqual(m.merge_starts(flags),[18,37])

    def test_truncated_baseline_tail(self):
        sd=np.array([.8,.1])
        self.assertTrue(m.candidates(sd)[0])
        self.assertFalse(m.candidates(sd,m.forward_complete(np.ones(2,bool))).any())

    def test_start_threshold(self):
        self.assertFalse(m.candidates(np.array([.4,.8,.0]))[0])

    def test_original_reference_equivalence(self):
        rng=np.random.default_rng(7)
        for _ in range(20):
            sd=rng.uniform(0,1,80);sd[rng.random(80)<.08]=np.nan
            expected=np.zeros(80,bool)
            for i in range(80):
                if not np.isfinite(sd[i]):continue
                w=sd[i:min(i+19,80)]
                if np.isfinite(w).sum()<2:continue
                peak,trough=int(np.nanargmax(w)),int(np.nanargmin(w))
                drop=w[peak]-w[trough]
                expected[i]=sd[i]>=.5 and np.isfinite(drop) and drop>=.25 and trough>peak
            np.testing.assert_array_equal(expected,m.candidates(sd))

    def test_complete_AB_identical_away_from_tail(self):
        d=fake_data()
        sd=m.row_std(d['raw']);gate=m.forward_complete(d['valid'].all(1))
        np.testing.assert_array_equal(m.candidates(sd)[:-18],m.candidates(sd,gate)[:-18])

    def test_core_subset_fixed(self):
        d=fake_data('yamanokami')
        core=m.take_depths(d,np.arange(1,11))
        np.testing.assert_allclose(core['depths'],np.arange(40,221,20)/100)
        self.assertEqual(core['raw'].shape[1],10)
        self.assertEqual(d['raw'].shape[1],11)

    def test_gmean_no_gap_bridging(self):
        out=m.gmean_adjacent(np.array([[0.,1.,np.nan,3.]]),np.array([0.,1.,2.,3.]))
        self.assertEqual(out[0],1.)
        out=m.gmean_adjacent(np.array([[0.,np.nan,2.]]),np.array([0.,1.,2.]))
        self.assertTrue(np.isnan(out[0]))

    def test_empty_catalog_has_columns(self):
        d=fake_data();table=m.make_catalog([],np.ones(90),'B',d,'original17')
        self.assertEqual(list(table.columns),m.EVENT_COLUMNS)
        self.assertEqual(len(m.audit_windows(table,table,np.zeros(90,bool))),0)

    def test_window_QC_provenance(self):
        d=fake_data();d['bad'][5,2]=True;d['valid'][5,2]=False
        sd=m.row_std(d['raw']);table=m.make_catalog([0],sd,'R0',d,'original17')
        self.assertEqual(int(table.iloc[0].QC_flagged_finite_cells),1)
        self.assertFalse(bool(table.iloc[0].strict_window_eligible))
        self.assertEqual(int(table.iloc[0].valid_depth_set_change_count),2)

    def test_read_schema(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'input.zip';zip_input(p)
            with zipfile.ZipFile(p) as z:
                self.assertEqual(m.read_site(z,'shin')['raw'].shape,(90,17))

    def test_mask_inconsistency_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'input.zip';zip_input(p,True)
            with zipfile.ZipFile(p) as z:
                with self.assertRaisesRegex(ValueError,'internally inconsistent'):
                    m.read_site(z,'shin')

    def test_two_site_end_to_end_synthetic(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'input.zip';zip_input(p)
            h=m.sha256(p)
            out=Path(td)/'out'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(m.run(p,out),0)
            self.assertEqual(m.sha256(p),h)
            b=json.loads((out/'batch_summary.json').read_text())
            self.assertEqual(b['status'],'completed')
            self.assertEqual(len(b['summaries']),3)
            self.assertTrue((out/'yamanokami/core10/events_D_same_full11_windows.csv').is_file())
            with zipfile.ZipFile(out/m.OUTPUT_NAME) as z:
                self.assertIsNone(z.testzip())
                self.assertIn('event_count_summary.csv',z.namelist())
            with self.assertRaises(FileExistsError):m.run(p,out)

    def test_explicit_input_finding(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.zip';p.write_bytes(b'x')
            self.assertEqual(m.find_input(Path(td),p),p.resolve())


if __name__=='__main__':
    unittest.main(verbosity=2)
