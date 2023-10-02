import unittest

import numpy as np
from GRF_metrics import GRF_metrics



class TestGRF(unittest.TestCase):
    def test_projection(self):
        """
        Test that the projection is calculated correctly.
        """
        x = np.array([(-4),(2),(7)])
        y = np.array([(3),(1),(2)])
        result = GRF_metrics.project_vector(x,y)
        np.testing.assert_array_equal(result, np.array([(6/7),(2/7),(4/7)]))


    def test_vectorAtoB(self):
        """
        Test that the vector substraction is working.
        """
        A = np.array([(1),(1),(2)])
        B = np.array([(1),(1),(1)])
        result = GRF_metrics.calculate_vec_AtoB(A,B)
        np.testing.assert_array_equal(result, np.array([(0),(0),(-1)]).reshape((3,1)))
        

if __name__ == '__main__':
    unittest.main()