import unittest
import os
import pandas as pd
from src.chant import Chant
from src.chant import get_chant_by_id

# Load demo chants
CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, os.path.pardir))
_demo_chants_fn = os.path.join(ROOT_DIR, 'cantus-data', 'chants-demo.csv')
CHANTS = pd.read_csv(_demo_chants_fn, index_col='id')

class TestChant(unittest.TestCase):

    def test_dummy(self):
        data = {
            'id': 'id1',
            'volpiano': 'abc--de-f'
        }
        chant = Chant(data)
        self.assertEqual(chant.id, 'id1')
        self.assertTrue(chant.has_volpiano)
    
    def test_init(self):
        chant = get_chant_by_id(CHANTS, CHANTS.index[0])
        self.assertEqual(chant.id, CHANTS.index[0])
        self.assertEqual(chant.get('id'), CHANTS.index[0])

