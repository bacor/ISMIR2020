import unittest
import doctest
from src import features
from src import filters
from src import representation
from src import segmentation
from src import volpiano

class TestDocTests(unittest.TestCase):
    def test_features(self):
        result = doctest.testmod(features)
        self.assertEqual(result.failed, 0)

    def test_filters(self):
        result = doctest.testmod(filters)
        self.assertEqual(result.failed, 0)

    def test_representation(self):
        result = doctest.testmod(representation)
        self.assertEqual(result.failed, 0)

    def test_segmentation(self):
        result = doctest.testmod(segmentation)
        self.assertEqual(result.failed, 0)

    def test_volpiano(self):
        result = doctest.testmod(volpiano)
        self.assertEqual(result.failed, 0)