import unittest
import os
import numpy as np
from src.segmentation import segment_poisson

class TestSegmentations(unittest.TestCase):

    def test_with_first_and_final_segment(self):
        np.random.seed(0)
        iterable = np.arange(20)
        segments = segment_poisson(iterable, lam=1)
        self.assertEqual(segments[0][0], 0)
        self.assertEqual(segments[-1][-1], 19)

    def test_large_lam(self):
        np.random.seed(0)
        iterable = np.arange(20)
        lam = 10 * len(iterable)

        # Test with the first and last: should be full iterable
        segments = segment_poisson(iterable, lam=lam)
        self.assertListEqual(list(segments[0]), list(iterable))
    
    def test_mean_close_to_lambda(self):
        np.random.seed(0)
        for lam in range (5, 15, 2):
            iterable = np.arange(10000)
            segments = segment_poisson(iterable, lam=lam)
            mean_len = np.mean([len(s) for s in segments])
            self.assertAlmostEqual(mean_len, lam, delta=.2)

