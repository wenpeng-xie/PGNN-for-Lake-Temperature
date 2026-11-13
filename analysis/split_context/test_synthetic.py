import unittest
import pandas as pd
import numpy as np
import summarize_split_context as m

class T(unittest.TestCase):
    def test_depth_parser(self):
        self.assertAlmostEqual(m.parse_depth_from_col("0.25m"),0.25)
        self.assertAlmostEqual(m.parse_depth_from_col("0.4 m"),0.4)
        self.assertAlmostEqual(m.parse_depth_from_col("temp_0.4m"),0.4)
        self.assertIsNone(m.parse_depth_from_col("0.4m_flag_nan"))

    def test_qstats(self):
        q1,med,q3=m.qstats([1,2,3,4])
        self.assertAlmostEqual(med,2.5)

    def test_native_selection(self):
        df=pd.DataFrame({
            "0.05m":[1],
            "0.25m":[2],
            "0.45m":[3],
            "2.25m":[4],
            "2.45m":[5],
        })
        parsed=m.native_temperature_columns(df,0.2,2.25)
        self.assertEqual([x[0] for x in parsed],[0.25,0.45,2.25])

if __name__=="__main__":
    unittest.main(verbosity=2)
