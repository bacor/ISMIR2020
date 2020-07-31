# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2019 Bas Cornelissen
# License: MIT
# -----------------------------------------------------------------------------
"""
Data generation
===============

This script reads out the raw Cantus data (not included in the repository) and
generates the datasets used in this study. We have different datasets for
different genres, and each comes in two variants (full and subset). For 
example:

* `responsory/full/`: a dataset of responsories 
* `responsory/subset/`: a subset where we made sure that every chant had a 
unique (cantus_id, mode) combination. This effectively means there are no (or 
in any case fewer) melody variants in the corpus.

The script is completely deterministic: all random seeds have been fixed. We 
also verified that two independent runs resulted in identical datasets (we log
md5 hashes of the generated files). However, all randomness is determined by 
one global random state, to be able to generate independent datasets for use
in independent runs.

Note that this script is a little slow. Generating the `responsory/full`
dataset for example takes around 2 minutes. (It could be further optimized, 
but there is no need to run it often.)

usage: `python -m src.generate_data [--what=demo/complete]`
"""
import os
import glob
import logging
import numpy as np
import pandas as pd

from .filters import *
from .segmentation import *
from .representation import *
from .volpiano import volpiano_characters
from .volpiano import expand_accidentals
from .volpiano import clean_volpiano
from .features import initial
from .features import final
from .features import lowest
from .features import highest
from .features import ambitus
from .features import initial_gesture
from .features import pitch_profile
from .features import pitch_class_profile
from .features import repetition_profile
from .helpers import md5checksum
from .helpers import relpath

CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, os.path.pardir))
CANTUS_DIR = os.path.join(ROOT_DIR, 'cantuscorpus', 'csv')

def save_csv(df: pd.DataFrame, filename: str, output_dir: str):
    """Stores a dataframe to a CSV file and logs its name and md5 hash

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe
    filename : str
        Filename
    output_dir : str
        Output directory
    """
    path = os.path.join(output_dir, filename)
    df.sort_index(inplace=True)
    df.to_csv(path)
    logging.info(f' > Stored to {relpath(path)}')
    logging.info(f'   md5 checksum: {md5checksum(path)}')

def load_chants(cantus_dir: str = CANTUS_DIR, demo: bool = False):
    """Load CantusCorpus chants data

    Parameters
    ----------
    cantus_dir : str
        The directory where all Cantus data is stored. Defaults to the 
        `cantus-data` directory in the root of this repository.
    demo : bool, default False
        Load the demo data? This is a small random subset of 100 chants
        for development purposes.

    Returns
    -------
    pd.DataFrame
        The chants
    """
    if demo:
        demo_fn = os.path.join(cantus_dir, 'chant-demo-sample.csv')
        chants = pd.read_csv(demo_fn, index_col='id')
    else:
        chants_fn = os.path.join(cantus_dir, 'chant.csv')
        chants = pd.read_csv(chants_fn, index_col='id')
    return chants

def filter_chants(chants: pd.DataFrame, genre: str,
    remove_duplicate_cantus_ids: bool) -> pd.DataFrame:
    """Filter out only complete, clean chants

    Parameters
    ----------
    chants : pd.DataFrame
        The chants dataframe
    genre : str
        The chant genre to include

    Returns
    -------
    pd.DataFrame
        A filtered dataframe of chants
    """
    opts = dict(logger=lambda msg: logging.info(f' . {msg}'))
    chants = filter_chants_without_volpiano(chants, **opts)
    chants = filter_chants_without_notes(chants, **opts)
    chants = filter_chants_without_simple_mode(
        chants, include_transposed=False, **opts)
    chants = filter_chants_without_full_text(chants, **opts)
    chants = filter_chants_where_incipit_is_full_text(chants, **opts)
    chants = filter_chants_by_genre(chants, include=[genre], **opts)
    
    chants = filter_chants_not_starting_with_G_clef(chants, **opts)
    chants = filter_chants_with_F_clef(chants, **opts)
    chants = filter_chants_with_missing_pitches(chants, **opts)
    chants = filter_chants_with_nonvolpiano_chars(chants, **opts)
    chants = filter_chants_without_word_boundary(chants, **opts)

    chants = filter_chants_with_duplicated_notes(chants, **opts)
    if remove_duplicate_cantus_ids:
        chants = sample_one_chant_per_mode_and_cantus_id(
            chants, random_state=np.random.randint(100), **opts)
    return chants

def process_volpiano(volpiano: str) -> str:
    """Clean up a volpiano string.

    First, it expands all accidentals, while omitting notes. That means that
    instead of for example `ij` for a b-flat, the accidental `i` is used to 
    represent the b-flat.
    
    Second, all characters that do not represent notes, liquescents, flats, 
    naturals, or boundaries (dashes) are removed. This results in a clean 
    volpiano string with only notes and boundaries.

    Parameters
    ----------
    volpiano : str
        A volpiano string

    Returns
    -------
    str
        The cleaned up volpiano string
    """
    volpiano = expand_accidentals(volpiano, omit_notes=True)
    chars = volpiano_characters('liquescents', 'notes', 'flats', 'naturals') + '-'
    volpiano = clean_volpiano(volpiano, allowed_chars=chars)
    return volpiano

def get_segmentations(chants: pd.DataFrame, k_min: int = 1, k_max: int = 16, 
                  sep: str = ' ') -> pd.DataFrame:
    """Generate all segmentations of the volpiano strings in a dataframe

    Parameters
    ----------
    chants : pd.DataFrame
        The chants dataframe with a `volpiano` column
    k_min : int, by default 1
        Length of the smallest k-mer segmentation
    k_max : int, by default 8
        Length of the largest k-mer segmentation
    sep : str, by default ' '
        The separator string to use

    Returns
    -------
    pd.DataFrame
        A dataframe with columns `words`, `syllables`, `neumes`, `k-mer` for
        each k_min <= k <= k_max, and `poisson` containing segmented volpiano
        strings.
    """
    # Helper function to join the segments
    join = lambda segments: sep.join(segments)

    # Clean up volpiano
    volpiano = chants.volpiano.map(process_volpiano)
    notes = volpiano.str.replace('-', '')
    
    # Natural segmentations
    df = pd.DataFrame(index=chants.index)
    df['words'] = volpiano.map(segment_words).map(join)
    df['syllables'] = volpiano.map(segment_syllables).map(join)
    df['neumes'] = volpiano.map(segment_neumes).map(join)
    
    # Baselines
    for k in range(k_min, k_max+1):
        kmer_segmenter = lambda vol: segment_kmers(vol, k=k)
        df[f'{k}-mer'] = notes.map(kmer_segmenter).map(join)
    poisson_segmenter_3 = lambda vol: segment_poisson(vol, lam=3)
    df['poisson-3'] = notes.map(poisson_segmenter_3).map(join)
    poisson_segmenter_5 = lambda vol: segment_poisson(vol, lam=5)
    df['poisson-5'] = notes.map(poisson_segmenter_5).map(join)
    poisson_segmenter_7 = lambda vol: segment_poisson(vol, lam=7)
    df['poisson-7'] = notes.map(poisson_segmenter_7).map(join)

    return df

def get_representation(segments, representation, dependent):
    """Convert the representation of segmented volpiano strings

    Parameters
    ----------
    segments : pd.DataFrame
        A dataframe where every column contains volpiano strings segmented by
        spaces.
    representation : { 'contour', 'interval' }
        The representation to convert to
    dependent : bool
        Convert to a dependent representation?

    Returns
    -------
    pd.DataFrame
        A dataframe of the same form, but now all volpiano strings have been
        converted to the target representation
    """
    df = pd.DataFrame(index=segments.index)

    if representation == 'contour':
        converter_fn = contour_representation
    elif representation == 'interval':
        converter_fn = interval_representation

    for col in segments.columns:
        values = segments[col]
        
        # Dependent representation:
        # Compute a relative representation of the entire chant directly,
        # but repeat the first note to ensure the length is the same.
        if dependent:
            kwargs = dict(repeat_first_note=True, first_interval_empty=False,
                          segment=True, sep=' ')
            converted = [converter_fn(volpiano, **kwargs) for volpiano in values]
        
        # Independent representation
        # First split the volpiano in units, and then compute the relative 
        # representation of each of the units.
        else:
            kwargs = dict(repeat_first_note=False, first_interval_empty=True, 
                          segment=True, sep=' ')
            converted = []
            for volpiano in values:
                units = volpiano.split(' ')
                conv_units = [converter_fn(unit, **kwargs) for unit in units]
                converted.append(' '.join(conv_units))
        df[col] = converted
    return df

def get_features(chants: pd.DataFrame) -> pd.DataFrame:
    """Extract features from a chants dataframe

    Parameters
    ----------
    chants : pd.DataFrame
        The chants dataframe

    Returns
    -------
    pd.DataFrame
        A dataframe with one feature per column
    """
    df = pd.DataFrame(index=chants.index)
    volpiano = chants.volpiano.map(process_volpiano)
    notes = volpiano.str.replace('-', '')
    pitches = notes.map(volpiano_to_midi)

    df['initial'] = pitches.map(initial)
    df['final'] = pitches.map(final)
    df['lowest'] = pitches.map(lowest)
    df['highest'] = pitches.map(highest)
    df['ambitus'] = pitches.map(ambitus)
    
    init_gestures = np.array(pitches.map(initial_gesture).to_list())
    for i in range(init_gestures.shape[1]):
        df[f'initial_gesture_{i+1}'] = init_gestures[:, i]

    pitch_profiles = np.array(pitches.map(pitch_profile).to_list())
    for i in range(pitch_profiles.shape[1]):
        df[f'freq_MIDI_{MIN_MIDI_PITCH+i}'] = pitch_profiles[:, i]

    pitch_class_profiles = np.array(pitches.map(pitch_class_profile).to_list())
    for i in range(pitch_class_profiles.shape[1]):
        df[f'freq_pitch_class_{i}'] = pitch_class_profiles[:, i]

    repetition_profiles = np.array(pitches.map(repetition_profile).to_list())
    for i in range(repetition_profiles.shape[1]):
        df[f'repetition_score_MIDI_{MIN_MIDI_PITCH+i}'] = repetition_profiles[:, i]

    return df

def generate_dataset(chants: pd.DataFrame, genre: str, id_pattern: str, 
                     output_dir: str, train_frac: float,
                     remove_duplicate_cantus_ids: bool,
                     random_state: int):
    """Generate a dataset of CSV files for one chant genre

    Parameters
    ----------
    chants : pd.DataFrame
        The full chants dataframe
    genre : str
        the genre to extract
    id_pattern : str
        A pattern for the new ids for the chants
    output_dir : str
        the output directory
    train_frac : float, by default 0.7
        The relative size of the training set, by default 0.7
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.basicConfig(
        filename=os.path.join(output_dir, 'data-generation.log'),
        filemode='w',
        format='%(levelname)s %(asctime)s %(message)s',
        datefmt='%d-%m-%y %H:%M:%S',
        level=logging.INFO)

    logging.info('*'*50)
    logging.info(f'* CREATING DATASET')
    logging.info(f'* genre: {genre}')
    logging.info(f'* train_frac: {train_frac}')
    logging.info(f'* remove_duplicate_cantus_ids: {remove_duplicate_cantus_ids}')
    logging.info(f'* output_dir: {relpath(output_dir)}')
    logging.info('*'*50)

    logging.info(f'Setting numpy random seed random_state={random_state}')
    np.random.seed(random_state)

    logging.info('Filtering chants...')
    chants = filter_chants(
        chants, genre, remove_duplicate_cantus_ids=remove_duplicate_cantus_ids)
    
    rs = np.random.randint(100)
    logging.info('Generate train and test split... (random_state={rs})')
    train = chants.sample(frac=train_frac, random_state=rs, replace=False)
    test_index = chants.index.difference(train.index)
    test = chants.loc[test_index, :].sample(frac=1, random_state=rs+1)
    logging.info(f'Split in training (N={len(train)}) and test (N={len(test)})')
    save_csv(test, f'test-chants.csv', output_dir)
    save_csv(train, f'train-chants.csv', output_dir)

    logging.info('Generate segmentations...')
    train_segm = get_segmentations(train)
    save_csv(train_segm, f'train-representation-pitch.csv', output_dir)
    test_segm = get_segmentations(test)
    save_csv(test_segm, f'test-representation-pitch.csv', output_dir)

    logging.info('Generate other representations...')
    for representation in ['interval', 'contour']:
        for dependent in [True, False]:
            dep = 'dependent' if dependent else 'independent'
            logging.info(f' {dep} {representation} representation')
            train_conv = get_representation(
                train_segm, representation=representation, dependent=dependent)
            save_csv(train_conv, f'train-representation-{representation}-{dep}.csv', output_dir)
            test_conv = get_representation(
                test_segm, representation=representation, dependent=dependent)
            save_csv(test_conv, f'test-representation-{representation}-{dep}.csv', output_dir)

    logging.info('Extract features...')
    train_feats = get_features(train)
    save_csv(train_feats, f'train-features.csv', output_dir)
    test_feats = get_features(test)
    save_csv(test_feats, f'test-features.csv', output_dir)
    logging.info('—'*50)

    # Close logging
    logger = logging.getLogger()
    logger.handlers[0].stream.close()
    logger.removeHandler(logger.handlers[0])

def generate_datasets(datasets: str, demo=False, output_dir='data',
    random_state=0):
    """Generate several datasets

    Parameters
    ----------
    datasets : list
        A list of dataset names to generate
    """
    chants = load_chants(CANTUS_DIR, demo=demo)
    
    if 'responsory-full' in datasets:
        generate_dataset(chants, 
            genre='genre_r', 
            id_pattern='resp{:0>5}', 
            output_dir=os.path.join(ROOT_DIR, output_dir, 'responsory', 'full'),
            remove_duplicate_cantus_ids=False,
            train_frac=0.7,
            random_state=random_state)

    if 'responsory-subset' in datasets:
        generate_dataset(chants, 
            genre='genre_r', 
            id_pattern='resp{:0>5}', 
            output_dir=os.path.join(ROOT_DIR, output_dir, 'responsory', 'subset'),
            remove_duplicate_cantus_ids=True,
            train_frac=0.7,
            random_state=random_state+1)

    if 'antiphon-subset' in datasets:
        generate_dataset(chants, 
            genre='genre_a', 
            id_pattern='anti{:0>5}', 
            output_dir=os.path.join(ROOT_DIR, output_dir, 'antiphon', 'subset'),
            remove_duplicate_cantus_ids=True,
            train_frac=0.7,
            random_state=random_state+2)
    
    if 'antiphon-full' in datasets:
        generate_dataset(chants, 
            genre='genre_a', 
            id_pattern='anti{:0>5}', 
            output_dir=os.path.join(ROOT_DIR, output_dir, 'antiphon', 'full'),
            remove_duplicate_cantus_ids=False,
            train_frac=0.7,
            random_state=random_state+3)
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--what', type=str, default='complete', 
        help='What to generate, the `complete` data, or `demo` data?')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    datasets = [
        'responsory-subset',
        'responsory-full',
        'antiphon-subset',
        'antiphon-full',
    ]
    if args.what == 'complete':
        generate_datasets(datasets, random_state=args.seed, 
            output_dir=os.path.join('data', f'run-{args.seed}'))
    elif args.what == 'demo':
        generate_datasets(datasets, demo=True, random_state=args.seed,
            output_dir=os.path.join('demo-data', f'run-{args.seed}'))

if __name__ == '__main__':
    main()