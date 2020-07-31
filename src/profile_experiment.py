# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2020 Bas Cornelissen
# License: MIT
# -----------------------------------------------------------------------------
"""
Profile experiment
======================

The code is completely deterministic. We have verified that multiple runs give
identical results (see e.g. the md5 hashes of predictions that are logged)
"""
import os
import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from .helpers import ROOT_DIR, DATA_DIR, RESULTS_DIR
from .helpers import GENRES, SUBSETS
from .helpers import start_experiment_log
from .helpers import relpath
from .classification import train_model
from .classification import get_scores

np.random.seed(0)

# Globals
# —————————————————————————————————————————————————————————————————————————————

PITCH_FEATURES = [f'freq_MIDI_{i}' for i in range(53, 87)]

PITCH_CLASS_FEATURES = [f'freq_pitch_class_{i}' for i in range(12)]

REPETITION_FEATURES = [f'repetition_score_MIDI_{i}' for i in range(53, 87)]

PROFILES = {
    'pitch': PITCH_FEATURES,
    'pitch_class': PITCH_CLASS_FEATURES,
    'repetition': REPETITION_FEATURES
}


# Helpers
# —————————————————————————————————————————————————————————————————————————————

def get_conditions(genres='all', subsets='all', profiles='all'):
    """Get a list of experimental conditions

    Parameters
    ----------
    genres : str or list, optional
        Genres to include, by default 'all'
    subsets : str or list, optional
        Subsets to include, by default 'all'
    profile : str or list, optional
        Profiles to include, by default 'all'

    Returns
    -------
    (list, dict)
        A list with all conditions, and a dictionary with keys 'genres', 
        'subsets' and 'profiles' containing those values.
    """
    subsets = SUBSETS if subsets == 'all' else subsets
    genres = GENRES if genres == 'all' else genres
    profiles = list(PROFILES.keys()) if profiles == 'all' else profiles
    
    conditions = []
    for genre in genres:
        for subset in subsets:
            for profile in profiles:
                conditions.append(
                    dict(genre=genre, subset=subset, profile=profile))

    parts = dict(subsets=subsets, genres=genres, profiles=profiles)
    return conditions, parts 

def load_dataset(genre, subset, profile, split, data_dir=DATA_DIR):
    """Load a dataset for training the classifier. Returns a dataframe of
    features and an array of corresponding targets (modes)"""
    feature_names = PROFILES[profile]
    features_fn = os.path.join(data_dir, genre, subset, f'{split}-features.csv')
    data = pd.read_csv(features_fn, index_col=0)[feature_names]
    chants_fn = os.path.join(data_dir, genre, subset, f'{split}-chants.csv')
    targets = pd.read_csv(chants_fn, index_col=0)['mode']
    assert len(targets) == len(data)
    return data, targets


# Experiment
# —————————————————————————————————————————————————————————————————————————————

def run_condition(genre, subset, profile,
                  data_dir, results_dir, 
                  n_iter, n_splits):
    """Runs a single experimental condition: trains the classifier, stores
    all the model, cross-validation results, and evaluation scores."""
    # Start experiment
    logging.info(f'Training model...')
    logging.info(f'* profile={profile}')
    logging.info(f'* genre={genre}')
    logging.info(f'* subset={subset}')

    # Set up directories
    output_dir = os.path.join(results_dir, genre, subset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load training and testing data
    kwargs = dict(genre=genre, subset=subset, profile=profile, data_dir=data_dir)
    train_data, train_targets = load_dataset(split='train', **kwargs)
    test_data, test_targets = load_dataset(split='test', **kwargs)
    logging.info(f'* Training/test size: {len(train_data)}/{len(test_data)}')
    logging.info(f'* Num. features: {train_data.shape[1]}')
    
    # Model parameters and param grid for tuning
    fixed_params = {
        'n_jobs': -1,
        'p': 2,
        'metric': 'minkowski'
    }
    tuned_params = {
        'n_neighbors': np.arange(1, 50),
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'leaf_size': np.arange(10, 100)
    }

    # Tune and train model
    model = KNeighborsClassifier(**fixed_params)
    train_model(
        model = model,
        train_data=train_data,
        train_targets=train_targets,
        test_data=test_data,
        test_targets=test_targets,
        param_grid=tuned_params,
        n_splits=n_splits,
        n_iter=n_iter,
        basepath=os.path.join(output_dir, profile)
    )

def run(experiment_name, description=None, 
        genres='all', subsets='all', profiles='all',
        n_iter=100, n_splits=5, 
        data_dir = DATA_DIR, results_dir = RESULTS_DIR):
    """Run a 'profile' mode classification experiment using pitch, pitch_class
    and repetition profiles."""
    # Set up directories
    results_dir = os.path.join(results_dir, experiment_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Get all conditions
    conditions, parts = get_conditions(genres, subsets, profiles)

    # Start log and log experiment settings
    start_experiment_log(
        name=experiment_name, 
        description=description,
        data_dir=data_dir,
        results_dir=results_dir,
        n_iter=n_iter,
        n_splits=n_splits,
        num_conditions=len(conditions),
        **parts)
    
    # Train all models
    for condition in conditions:
        run_condition(
            data_dir=data_dir, results_dir=results_dir,
            n_iter=n_iter, n_splits=n_splits, **condition)

def evaluate(experiment_name, 
             genres='all', subsets='all', profiles='all',
             data_dir = DATA_DIR, results_dir = RESULTS_DIR, **kwargs):
    """Evaluate an experiment and store accuracy and retrieval scores in a
    single CSV file that can be used to e.g. generate tables and figures."""
    logging.info('Evaluating experiment...')
    results_dir = os.path.join(results_dir, experiment_name)
    scores = []
    conditions, _ = get_conditions(genres, subsets, profiles)
    for condition in conditions:
        profile = condition['profile']
        output_dir = os.path.join(
            results_dir, condition['genre'], condition['subset'])
        condition_scores = get_scores(
            test_pred_fn = os.path.join(output_dir, f'{profile}-test-pred.txt'),
            train_pred_fn = os.path.join(output_dir, f'{profile}-train-pred.txt'),
            genre=condition['genre'],
            subset=condition['subset'],
            data_dir=data_dir)
        condition_scores.update(condition)
        scores.append(condition_scores)
        
    scores_fn = os.path.join(results_dir, f'{experiment_name}-scores.csv')
    pd.DataFrame(scores).to_csv(scores_fn)
    logging.info(f'> Stored scores to {relpath(scores_fn)}')
