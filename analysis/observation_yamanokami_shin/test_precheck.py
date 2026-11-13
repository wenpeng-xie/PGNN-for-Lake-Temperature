"""Small synthetic-data tests only; do not load the user's real observations."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
import check_observation_completeness as m


def fixture(depths=(0.25, 0.45), n=40, spaced=False):
    t = pd.date_range('2025-01-01', periods=n, freq='10min')
    f = pd.DataFrame({'time': t})
    for d in depths:
        c = f'{d:.1f} m' if spaced else f'{d:.2f}m'
        f[c] = 10.0 + d
        for flag in m.DATA_FLAGS + m.IGNORED_FLAGS:
            f[f'{c}_{flag}'] = False
    return f


def mini_config(filename='input.csv', depths=(0.25, 0.45)):
    return dict(name='synthetic', file=filename, depths=list(depths), start='2025-01-01 00:00:00',
                observation_end='2025-01-01 06:30:00', joint_end='2025-01-01 05:50:00', period_note='unit test only')


class PrecheckTests(unittest.TestCase):
    def test_fixed_site_counts_and_dates(self):
        self.assertEqual(len(m.SITES['shin']['depths']), 17)
        self.assertEqual(m.SITES['shin']['depths'][-1], 3.45)
        self.assertEqual(len(m.SITES['yamanokami']['depths']), 11)
        self.assertEqual(m.SITES['yamanokami']['depths'][0], 0.20)
        self.assertEqual(m.SITES['yamanokami']['depths'][-1], 2.20)
        for cfg in m.SITES.values():
            self.assertLess(pd.Timestamp(cfg['start']), pd.Timestamp(cfg['joint_end']))
            self.assertLessEqual(pd.Timestamp(cfg['joint_end']), pd.Timestamp(cfg['observation_end']))

    def test_exact_depth_format_mapping(self):
        frame = fixture((0.2, 0.4), spaced=True)
        tc, cols, req, opt = m.resolve_columns(list(frame.columns), [0.20, 0.40])
        self.assertEqual(tc, 'time')
        self.assertEqual(cols, ['0.2 m', '0.4 m'])
        self.assertEqual(len(req), 8)
        self.assertEqual(len(opt), 6)

    def test_missing_qc_column_not_accepted(self):
        cols = list(fixture().columns)
        cols.remove('0.25m_flag_spike')
        with self.assertRaises(ValueError): m.resolve_columns(cols, [0.25, 0.45])

    def test_no_nearest_depth_substitution(self):
        with self.assertRaises(ValueError): m.resolve_columns(list(fixture().columns), [0.20, 0.40])

    def test_duplicate_numeric_depth_rejected(self):
        f = fixture()
        f['0.250 m'] = 10.0
        with self.assertRaises(ValueError): m.resolve_columns(list(f.columns), [0.25, 0.45])

    def test_flags_literal_strings(self):
        s = pd.Series(['False', 'True', '0', '1', '0.0', '1.0', 'f', 't'])
        bad, known = m.parse_flag(s, 'test')
        self.assertTrue(known.all())
        np.testing.assert_array_equal(bad, [False, True]*4)

    def test_unknown_flag_is_not_good(self):
        bad, known = m.parse_flag(pd.Series([False, None, '', np.nan]), 'test')
        np.testing.assert_array_equal(known, [True, False, False, False])
        self.assertFalse(bad.any())

    def test_unexpected_flags_fail(self):
        with self.assertRaises(ValueError): m.parse_flag(pd.Series(['maybe']), 'test')

    def test_time_missing_row_inserted_not_compacted(self):
        f = fixture().drop(index=10)
        t, d, present, meta = m.aligned_frame(f, 'time', '2025-01-01', '2025-01-01 06:30:00')
        self.assertEqual(len(t), 40)
        self.assertFalse(present[10])
        self.assertTrue(d.iloc[10].isna().all())
        self.assertEqual(meta['source_rows_in_target_period'], 39)

    def test_duplicate_time_rejected(self):
        f = fixture()
        f = pd.concat([f, f.iloc[[1]]], ignore_index=True)
        with self.assertRaises(ValueError): m.aligned_frame(f, 'time', '2025-01-01', '2025-01-01 06:30:00')

    def test_offgrid_rejected(self):
        f = fixture(); f.loc[1, 'time'] += pd.Timedelta(seconds=1)
        with self.assertRaises(ValueError): m.aligned_frame(f, 'time', '2025-01-01', '2025-01-01 06:30:00')

    def test_timezone_rejected(self):
        f = fixture(); f['time'] = pd.to_datetime(f['time']).dt.tz_localize('Asia/Tokyo')
        with self.assertRaises(ValueError): m.aligned_frame(f, 'time', '2025-01-01', '2025-01-01 06:30:00')

    def test_window_19_endpoints_and_tail(self):
        x = np.ones(40, dtype=bool)
        r = m.forward_all(x)
        self.assertEqual(int(r.sum()), 22)
        self.assertTrue(r[:22].all())
        self.assertFalse(r[22:].any())
        x[18] = False
        r = m.forward_all(x)
        self.assertFalse(r[:19].any())
        self.assertTrue(r[19:22].all())

    def test_short_window_no_eligible_starts(self):
        self.assertFalse(m.forward_all(np.ones(18, dtype=bool)).any())
        self.assertEqual(m.forward_all(np.ones(19, dtype=bool)).tolist(), [True]+[False]*18)

    def test_same_count_changed_depths(self):
        t = pd.date_range('2025-01-01', periods=40, freq='10min')
        raw = np.ones((40, 2))
        valid = np.ones((40, 2), dtype=bool); valid[0, 0] = False; valid[1, 1] = False
        summary = m.completeness_summary(t, raw, valid, np.ones(40, bool), ~valid, np.zeros_like(valid))
        self.assertEqual(summary['n_depth_set_changes_with_same_layer_count'], 1)
        self.assertEqual(summary['decision_code'], 'FOLLOWUP_FIXED_DEPTH_ANALYSIS_NEEDED')

    def test_gap_intervals_edges(self):
        self.assertEqual(m.true_intervals(np.array([1, 1, 0, 1, 0, 1], dtype=bool)), [(0, 1), (3, 3), (5, 5)])
        self.assertEqual(m.true_intervals(np.zeros(4, bool)), [])
        self.assertEqual(m.true_intervals(np.ones(4, bool)), [(0, 3)])

    def test_month_crossing_window_not_reset(self):
        t = pd.date_range('2025-01-31 22:00', periods=40, freq='10min')
        raw = np.ones((40, 2)); valid = np.ones((40, 2), bool)
        rows = m.monthly_rows('test', t, raw, valid, np.ones(40, bool))
        self.assertEqual(sum(r['complete_3h_candidate_starts'] for r in rows), 22)
        self.assertEqual(rows[0]['complete_3h_candidate_starts'], 12)

    def run_synthetic_site(self, modifier=None):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        root = Path(tmp.name); f = fixture()
        if modifier: f = modifier(f)
        p = root / 'input.csv'; f.to_csv(p, index=False)
        before = p.read_bytes()
        result = m.run_site(root, mini_config(), root / 'result')
        self.assertEqual(before, p.read_bytes())
        return root, result

    def test_no_qc_missing_skip_only_obs_contrast(self):
        root, s = self.run_synthetic_site()
        joint = s['scopes']['joint_evaluation_span']
        self.assertEqual(joint['decision_code'], 'NO_OBSERVATION_MISSINGNESS_CONTRAST_NEEDED')
        self.assertEqual(joint['n_expected_times'], 36)
        self.assertEqual(joint['n_full_3h_candidate_starts'], 18)
        self.assertEqual(joint['n_tail_points_without_full_3h'], 18)
        self.assertTrue(s['no_model_prediction_check'])
        self.assertTrue(s['no_event_detection'])
        with np.load(root/'result/observations_for_followup.npz', allow_pickle=False) as z:
            self.assertEqual(z['temperature_raw'].shape, (40, 2))
            self.assertTrue(z['valid_data_only'].all())

    def test_qc_valid_numerical_zeros_not_removed(self):
        def modify(f): f['0.25m'] = 0.0; return f
        _, s = self.run_synthetic_site(modify)
        self.assertTrue(s['scopes']['joint_evaluation_span']['all_nominal_depths_complete'])

    def test_physics_only_flags_not_removed(self):
        def modify(f):
            f['0.25m_flag_unstable'] = True
            f['0.45m_flag_vert'] = True
            f['0.25m_qc_bad'] = True
            return f
        _, s = self.run_synthetic_site(modify)
        self.assertTrue(s['scopes']['joint_evaluation_span']['all_nominal_depths_complete'])

    def test_qc_bad_counts_even_if_no_nan(self):
        def modify(f): f.loc[12, '0.25m_flag_stuck'] = True; return f
        _, s = self.run_synthetic_site(modify)
        joint = s['scopes']['joint_evaluation_span']
        self.assertEqual(joint['n_complete_raw_profiles'], 36)
        self.assertEqual(joint['n_complete_data_quality_profiles'], 35)
        self.assertEqual(joint['n_finite_known_bad_qc_cells'], 1)
        self.assertFalse(joint['all_nominal_depths_complete'])

    def test_missing_row_counts_and_intervals(self):
        _, s = self.run_synthetic_site(lambda f: f.drop(index=10))
        q = s['scopes']['joint_evaluation_span']
        self.assertEqual(q['n_absent_timestamp_rows'], 1)
        self.assertEqual(q['n_complete_data_quality_profiles'], 35)
        self.assertEqual(q['n_finite_unknown_qc_cells'], 0)
        self.assertEqual(q['n_finite_known_bad_qc_cells'], 0)

    def test_unknown_finite_flags_recorded(self):
        def modify(f):
            f['0.25m_flag_range'] = f['0.25m_flag_range'].astype(object)
            f.loc[5, '0.25m_flag_range'] = None
            return f
        _, s = self.run_synthetic_site(modify)
        self.assertEqual(s['scopes']['joint_evaluation_span']['n_finite_unknown_qc_cells'], 1)

    def test_tail_gap_not_used_to_shrink_period(self):
        def modify(f): f.loc[39, '0.25m'] = np.nan; return f
        _, s = self.run_synthetic_site(modify)
        self.assertTrue(s['scopes']['joint_evaluation_span']['all_nominal_depths_complete'])
        self.assertFalse(s['scopes']['observation_target_span']['all_nominal_depths_complete'])

    def test_rerun_does_not_overwrite_site_result(self):
        root, s = self.run_synthetic_site()
        with self.assertRaises(FileExistsError): m.run_site(root, mini_config(), root/'result')

    def test_batch_package_errors_are_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fixture().to_csv(root/'ok.csv', index=False)
            sites = {'shin': mini_config('ok.csv'), 'yamanokami': mini_config('absent.csv')}
            sites['shin']['name'] = 'Shin'; sites['yamanokami']['name'] = 'Yamanokami'
            with patch.object(m, 'SITES', sites), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = m.main(['--root', str(root)])
            self.assertEqual(code, 2)
            archives = list((root/'revision_R1_5').rglob('Shin_Yamanokami_R1_5_precheck_return.zip'))
            self.assertEqual(len(archives), 1)
            with zipfile.ZipFile(archives[0]) as z:
                self.assertIsNone(z.testzip())
                b = json.loads(z.read('batch_summary.json'))
                self.assertEqual(b['status'], 'incomplete_check_errors')
                self.assertEqual(b['completed_sites'], ['Shin'])
                self.assertEqual(len(json.loads(z.read('RUN_ERRORS.json'))), 1)
                self.assertIn('shin/summary.json', z.namelist())
                self.assertIn('REPORT_zh.md', z.namelist())

    def test_two_sites_one_command(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fixture().to_csv(root/'s.csv', index=False)
            fixture((0.2, 0.4), spaced=True).to_csv(root/'y.csv', index=False)
            sites = {'shin': mini_config('s.csv'), 'yamanokami': mini_config('y.csv', (0.2, 0.4))}
            sites['shin']['name'] = 'Shin'; sites['yamanokami']['name'] = 'Yamanokami'
            with patch.object(m, 'SITES', sites), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(m.main(['--root', str(root)]), 0)
            archive = next((root/'revision_R1_5').rglob('Shin_Yamanokami_R1_5_precheck_return.zip'))
            with zipfile.ZipFile(archive) as z:
                b = json.loads(z.read('batch_summary.json'))
                self.assertEqual(b['status'], 'completed')
                self.assertEqual(len(b['completed_sites']), 2)
                self.assertIn('yamanokami/coverage_by_month.csv', z.namelist())


if __name__ == '__main__':
    unittest.main(verbosity=2)
