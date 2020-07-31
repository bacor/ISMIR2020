# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2020 Bas Cornelissen
# -----------------------------------------------------------------------------
"""
Traditional experiment
======================

The code is completely deterministic. We have verified that multiple runs give
identical results (see e.g. the md5 hashes of predictions that are logged,
of look at commit 88e5b9d1f49fae76bad4c68294669665a5a03a71 which does not 
include any changes to the predictions)
"""
import os
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .helpers import save_model
from .helpers import ROOT_DIR, DATA_DIR, RESULTS_DIR
from .helpers import GENRES, SUBSETS
from .helpers import read_predictions
from .helpers import relpath
from .helpers import start_experiment_log
from .classification import train_model
from .classification import get_scores


# Globals
# —————————————————————————————————————————————————————————————————————————————

FEATURE_SETS = {
    'final': ['final'],
    'ambitus': ['lowest', 'highest'],
    'initial': ['initial'],
    'final-ambitus': ['final', 'lowest', 'highest'],
    'final-initial': ['final', 'initial'],
    'final-ambitus-initial': ['final', 'lowest', 'highest', 'initial'],
    'ambitus-initial': ['lowest', 'highest', 'initial']
}


# Helpers
# —————————————————————————————————————————————————————————————————————————————

def get_conditions(genres='all', subsets='all', feature_sets='all'):
    """Get a list of experimental conditions

    Parameters
    ----------
    genres : str or list, optional
        Genres to include, by default 'all'
    subsets : str or list, optional
        Subsets to include, by default 'all'
    feature_sets : str or list, optional
        Feature sets to include, by default 'all'

    Returns
    -------
    (list, dict)
        A list with all conditions, and a dictionary with keys 'genres', 
        'subsets' and 'feature_sets' containing those values.
    """
    subsets = SUBSETS if subsets == 'all' else subsets
    genres = GENRES if genres == 'all' else genres
    feature_sets = list(FEATURE_SETS.keys()) if feature_sets == 'all' else feature_sets
    
    conditions = []
    for genre in genres:
        for subset in subsets:
            for feature_set in feature_sets:
                conditions.append(
                    dict(genre=genre, subset=subset, feature_set=feature_set))

    parts = dict(subsets=subsets, genres=genres, feature_sets=feature_sets)
    return conditions, parts 

def load_dataset(genre, subset, feature_set, split, data_dir=DATA_DIR):
    """Load a dataset for training the classifier. Returns a dataframe of
    features and an array of corresponding targets (modes)"""
    feature_names = FEATURE_SETS[feature_set]
    features_fn = os.path.join(data_dir, genre, subset, f'{split}-features.csv')
    data = pd.read_csv(features_fn, index_col=0)[feature_names]
    chants_fn = os.path.join(data_dir, genre, subset, f'{split}-chants.csv')
    targets = pd.read_csv(chants_fn, index_col=0)['mode']
    assert len(targets) == len(data)
    return data, targets


# Experiment
# —————————————————————————————————————————————————————————————————————————————

def run_condition(feature_set, genre, subset, 
                  data_dir, results_dir, 
                  n_iter, n_splits):
    """Runs a single experimental condition: trains the classifier, stores
    all the model, cross-validation results, and evaluation scores."""
    # Start experiment
    logging.info(f'Training model...')
    logging.info(f'* feature_set={feature_set}')
    logging.info(f'* genre={genre}')
    logging.info(f'* subset={subset}')

    # Set up directories
    output_dir = os.path.join(results_dir, genre, subset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load training and testing data
    kwargs = dict(genre=genre, subset=subset, feature_set=feature_set, data_dir=data_dir)
    train_data, train_targets = load_dataset(split='train', **kwargs)
    test_data, test_targets = load_dataset(split='test', **kwargs)
    logging.info(f'* Training/test size: {len(train_data)}/{len(test_data)}')
    logging.info(f'* Num. features: {train_data.shape[1]}')
    
    # Model parameters and param grid for tuning
    fixed_params = {
        'random_state': np.random.randint(100),
        'n_jobs': -1,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0
    }
    tuned_params = {
        'criterion': ['gini', 'entropy'],
        'n_estimators': np.arange(100, 1000),
        'max_depth': np.arange(1, 1000),
        'max_features': np.linspace(0, 1, 100),
        'bootstrap': [True, False],
        'max_samples': np.linspace(.5, 1, 100)
    }

    # Train and store the model
    model = RandomForestClassifier(**fixed_params)
    train_model(
        model = model,
        train_data=train_data,
        train_targets=train_targets,
        test_data=test_data,
        test_targets=test_targets,
        param_grid=tuned_params,
        n_splits=n_splits,
        n_iter=n_iter,
        basepath=os.path.join(output_dir, feature_set)
    )
    
def run(experiment_name, description=None, 
        genres='all', subsets='all', feature_sets='all',
        n_iter=100, n_splits=5, 
        data_dir = DATA_DIR, results_dir = RESULTS_DIR):
    """Run a 'traditional' mode classification experiment using features such
    as final, ambitus and initial."""
    # Set up directories
    results_dir = os.path.join(results_dir, experiment_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Get all conditions
    conditions, parts = get_conditions(genres, subsets, feature_sets)

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
             genres='all', subsets='all', feature_sets='all',
             data_dir = DATA_DIR, results_dir = RESULTS_DIR, **kwargs):
    """Evaluate an experiment and store accuracy and retrieval scores in a
    single CSV file that can be used to e.g. generate tables and figures."""
    logging.info('Evaluating experiment...')
    results_dir = os.path.join(results_dir, experiment_name)
    conditions, _ = get_conditions(genres, subsets, feature_sets)
    scores = []
    for condition in conditions:
        feature_set = condition['feature_set']
        output_dir = os.path.join(
            results_dir, condition['genre'], condition['subset'])
        condition_scores = get_scores(
            test_pred_fn = os.path.join(output_dir, f'{feature_set}-test-pred.txt'),
            train_pred_fn = os.path.join(output_dir, f'{feature_set}-train-pred.txt'),
            genre=condition['genre'],
            subset=condition['subset'],
            data_dir=data_dir)
        condition_scores.update(condition)
        scores.append(condition_scores)
    scores_fn = os.path.join(results_dir, f'{experiment_name}-scores.csv')
    pd.DataFrame(scores).to_csv(scores_fn)
    logging.info(f'> Stored scores to {relpath(scores_fn)}')