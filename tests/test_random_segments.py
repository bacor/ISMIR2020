# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2020 Bas Cornelissen
# License: 
# -------------------------------------------------------------------
"""
"""
import unittest
import os
import numpy as np
from src.random_segments import poisson_segmentation
from src.random_segments import extract_random_segments_from_file

class TestPoissonSegmentation(unittest.TestCase):
    
    def test_with_first_and_final_segment(self):
        np.random.seed(0)
        iterable = np.arange(20)
        segments = poisson_segmentation(iterable, lam=1, 
            omit_first_and_last=False)
        self.assertEqual(segments[0][0], 0)
        self.assertEqual(segments[-1][-1], 19)

    def test_large_lam(self):
        np.random.seed(0)
        iterable = np.arange(20)
        lam = 10 * len(iterable)

        # Test with the first and last: should be full iterable
        segments = poisson_segmentation(iterable, lam=lam, 
            omit_first_and_last=False)
        self.assertListEqual(list(segments[0]), list(iterable))

        # Test without first and last: should be empty
        segments = poisson_segmentation(iterable, lam=lam, 
            omit_first_and_last=True)
        self.assertEqual(len(segments), 0)
    
    def test_without_first_and_final_segments(self):
        np.random.seed(0)
        iterable = np.arange(20)
        segments = poisson_segmentation(iterable, lam=1, omit_first_and_last=True)
        self.assertNotEqual(segments[0][0], 0)
        self.assertNotEqual(segments[-1][-1], 19)

    def test_mean_close_to_lambda(self):
        np.random.seed(0)
        for lam in range (5, 15, 2):
            iterable = np.arange(10000)
            segments = poisson_segmentation(iterable, lam=lam, omit_first_and_last=True)
            mean_len = np.mean([len(s) for s in segments])
            self.assertAlmostEqual(mean_len, lam, delta=.2)


class TestKernSegmentExtraction(unittest.TestCase):

    def test_extract_from_file(self):
        cur_dir = os.path.dirname(__file__)
        path = os.path.join(cur_dir, 'test.krn')
        np.random.seed(0)
        segments = extract_random_segments_from_file(path, lam=2,
            omit_first_and_last=False)

        notes = 'CDFGAAAGFDCCD'
        i = 0
        for segment in segments:
            for note in segment:
                self.assertEqual(note.name, notes[i])
                i += 1

if __name__ == '__main__':
    unittest.main()    