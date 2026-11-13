import unittest
import numpy as np
import update_fixed_depth_stability as m

class T(unittest.TestCase):
    def test_density_near_four(self):
        rho4 = float(m.water_density_freshwater(np.array([4.0]))[0])
        rho20 = float(m.water_density_freshwater(np.array([20.0]))[0])
        self.assertGreater(rho4, rho20)

    def test_bathymetry(self):
        d,a=m.build_bathymetry(m.SITE_CONFIGS["shiode"],dz=0.01)
        self.assertAlmostEqual(float(d[0]),0.0)
        self.assertAlmostEqual(float(d[-1]),2.45,places=6)
        self.assertGreaterEqual(float(a.min()),0.0)

    def test_thermocline(self):
        z=np.array([0.4,0.6,0.8,1.0])
        t=np.array([25.,25.,20.,20.])
        zt=m.thermocline_depth_from_sensors(z,t)
        self.assertAlmostEqual(zt,0.7)

    def test_wind(self):
        u=m.convert_wind_to_10m(2.0,z=2.0)
        self.assertGreater(u,2.0)

if __name__=="__main__":
    unittest.main(verbosity=2)
