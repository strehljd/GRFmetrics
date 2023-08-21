import unittest

import numpy as np
from GRF_metrics import GRF_metrics



class TestGRF(unittest.TestCase):
    def test_list_int(self):
        """
        Test that it can sum a list of integers
        """
        x = np.array([(-4),(2),(7)])
        y = np.array([(3),(1),(2)])
        result = GRF_metrics.project_vector(x,y)
        np.testing.assert_array_equal(result, np.array([(6/7),(2/7),(4/7)]))

        

if __name__ == '__main__':
    unittest.main()