import unittest
import numpy as np
import pandas as pd
import run_threshold_sensitivity as m

class T(unittest.TestCase):
    def test_forward_complete(self):
        x=np.ones(50,dtype=bool)
        y=m.forward_complete(x)
        self.assertTrue(y[:32].all())
        self.assertFalse(y[32:].any())

    def test_detect_one(self):
        sd=np.ones(60)*0.6
        # Create one decline in first 3h.
        sd[:19]=np.linspace(0.8,0.4,19)
        eligible=np.ones(60,dtype=bool)
        ev,c=m.detect_events(sd,eligible,0.5,0.25)
        self.assertGreaterEqual(len(ev),1)
        self.assertTrue((ev.obs_amplitude_degC>=0.25).all())

    def test_incomplete_gate(self):
        sd=np.linspace(0.8,0.3,40)
        eligible=np.zeros(40,dtype=bool)
        ev,c=m.detect_events(sd,eligible,0.5,0.25)
        self.assertEqual(len(ev),0)

    def test_std(self):
        a=np.tile(np.arange(4,dtype=float),(5,1))
        s=m.row_std(a)
        self.assertEqual(len(s),5)
        self.assertTrue(np.allclose(s,s[0]))

if __name__=="__main__":
    unittest.main(verbosity=2)
