# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2020 Bas Cornelissen
# -----------------------------------------------------------------------------
"""
TF-IDF based mode classification
================================

This module runs the mode classification experiment using tf-idf 
representations of the chants.

The code is completely deterministic: all random seeds have been fixed and we
have verified that independent runs yield identical results by checking the
md5 checksums of the train and test predictions (the serialized model and 
tunings results have different checksums, perhaps due to e.g. different fit 
times)
"""
import os
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.fixes import loguniform

from .helpers import ROOT_DIR, DATA_DIR, RESULTS_DIR
from .helpers import GENRES, SUBSETS
from .helpers import relpath
from .helpers import param_str
from .helpers import start_experiment_log
from .classification import train_model
from .classification import get_scores

np.random.state(0)

# Globals
# —————————————————————————————————————————————————————————————————————————————

CLASSIFIERS = ['linear_svc']

SEGMENTATIONS = [
    'neumes', 
    'syllables', 
    'words', 
    '1-mer', 
    '2-mer', 
    '3-mer', 
    '4-mer', 
    '5-mer', 
    '6-mer', 
    '7-mer', 
    '8-mer',
    '9-mer',
    '10-mer',
    '11-mer',
    '12-mer',
    '13-mer',
    '14-mer',
    '15-mer',
    '16-mer',
    'poisson-3',
    'poisson-5',
    'poisson-7',
]

REPRESENTATIONS = [
    'pitch',
    'interval-dependent',
    'interval-independent',
    'contour-dependent',
    'contour-independent'
]

# Helpers
# —————————————————————————————————————————————————————————————————————————————

def get_conditions(classifiers='all', genres='all', subsets='all', 
                   representations='all', segmentations='all'):
    """Get a list of experimental conditions"""
    classifiers = CLASSIFIERS if classifiers == 'all' else classifiers
    genres = GENRES if genres == 'all' else genres
    subsets = SUBSETS if subsets == 'all' else subsets
    representations = REPRESENTATIONS if representations == 'all' else representations
    segmentations = SEGMENTATIONS if segmentations == 'all' else segmentations
    
    conditions = []
    for classifier in classifiers:
        for genre in genres:
            for subset in subsets:
                for rep in representations:
                    for segm in segmentations:
                        conditions.append(dict(
                            classifier=classifier,
                            genre=genre, 
                            subset=subset, 
                            representation=rep,
                            segmentation=segm))

    parts = dict(classifiers=classifiers, genres=genres, subsets=subsets, 
                 representations=representations, segmentations=segmentations)
    return conditions, parts 

def load_dataset(genre, subset, representation, segmentation, split, 
                 data_dir=DATA_DIR):
    """Load a training dataset for mode classification, with the target modes

    Parameters
    ----------
    genre : str
        The genre, e.g. 'antiphon' or 'responsory'
    subset : str
        Full or subset?
    Split : { 'train', 'test' }
        Which subset to load: train or test set?
    representation : str
        The representation (pitch, contour-independent, etc.)
    segmentation : str
        The segmentation: words, syllables, k-mer, poisson-k.
    data_dir : str, optional
        The directory containing all data, by default the `data` directory in 
        the root of this repository.

    Returns
    -------
    (train_data, targets)
        The training data and targets as a pd.DataFrame and pd.Series
    """
    targets_fn = os.path.join(data_dir, genre, subset, f'{split}-chants.csv')
    targets = pd.read_csv(targets_fn, index_col=0)['mode']

    data_fn = os.path.join(
        data_dir, genre, subset, f'{split}-representation-{representation}.csv')
    data = pd.read_csv(data_fn, index_col=0)[segmentation]
    
    assert len(data) == len(targets)
    return data, targets


# Experiment
# —————————————————————————————————————————————————————————————————————————————

def get_vectorizer():
    """Get the standard tfidf-vectorizer"""
    tfidf_params = dict(
        # Defaults
        strip_accents=None,
        stop_words=None,
        ngram_range=(1,1),

        # Important: if you have a max_df strictly below 1, conditions with 
        # small vocabularies can end up with empty feature sets. That's why
        # we fix max_df, rather than tuning it. With a grid-search, you could
        # also tune it, but using randomized search (as we do), chances that 
        # you fail to test max_df=1.0 are too high, and the model won't fit.
        max_df=1.0,

        # Similarly, for datasets with high ttr (nearly all tokens are unique),
        # Setting min_df > 0.0 will result in very small or empty vocabularies.
        # For that reason, set min_df=1 (the default)
        min_df=1,

        # Tuning this seems to have surprisingly little effect
        max_features=5000,

        # These had virtually no effect on performance during tuning, so we
        # fix these to the defaults. This means that the vectorization is 
        # identical for all our models.
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,

        # Notes and liquescents are distinguished in volpiano by their case
        lowercase=False,

        # This is crucial, as the default tokenization is inappropriate for
        # our data
        analyzer='word',
        token_pattern=r'[^ ]+')
    logging.info(f'> vectorizer: {param_str(tfidf_params)}') 
    vectorizer = TfidfVectorizer(**tfidf_params)
    return vectorizer

def run_condition(classifier, genre, subset, representation, segmentation,
                  data_dir, results_dir, 
                  n_iter, n_splits):
    """Runs a single experimental condition: trains the classifier, stores
    all the model, cross-validation results, and evaluation scores."""
    # Start experiment
    logging.info(f'Training model...')
    logging.info(f'* classifier={classifier}')
    logging.info(f'* genre={genre}')
    logging.info(f'* subset={subset}')
    logging.info(f'* representation={representation}')
    logging.info(f'* segmentation={segmentation}')

    # Set up directories
    output_dir = os.path.join(results_dir, classifier, genre, subset, representation)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load training and testing data
    kwargs = dict(genre=genre, subset=subset, representation=representation, 
                  segmentation=segmentation, data_dir=data_dir)
    train_data, train_targets = load_dataset(split='train', **kwargs)
    test_data, test_targets = load_dataset(split='test', **kwargs)
    logging.info(f'* Training/test size: {len(train_data)}/{len(test_data)}')
    
    # Linear SVC
    if classifier == 'linear_svc':
        svc_params = {
            'penalty': 'l2',
            'loss': 'squared_hinge',
            'multi_class': 'ovr',
            'random_state': np.random.randint(100)
        }
        tuned_params = {
            'clf__C': loguniform(1e-3, 1e4),
            'clf__dual': [True, False]
        }
        model = Pipeline([
            ('vect', get_vectorizer()),
            ('clf', LinearSVC(**svc_params)),
        ])

    # Others
    else:
        raise ValueError(f'Classifier {classifier} is not yet supported')

    # Train and store the model
    train_model(
        model = model,
        train_data=train_data,
        train_targets=train_targets,
        test_data=test_data,
        test_targets=test_targets,
        param_grid=tuned_params,
        n_splits=n_splits,
        n_iter=n_iter,
        basepath=os.path.join(output_dir, segmentation)
    )

def run(experiment_name, description=None, 
        classifiers='all', genres='all', subsets='all', 
        representations='all', segmentations='all',
        n_iter=100, n_splits=5, 
        data_dir = DATA_DIR, results_dir = RESULTS_DIR):
    """Run a 'profile' mode classification experiment using pitch, pitch_class
    and repetition profiles."""
    # Set up directories
    results_dir = os.path.join(results_dir, experiment_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Get all conditions
    conditions, parts = get_conditions(
        classifiers, genres, subsets, representations, segmentations)

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
             classifiers='all', genres='all', subsets='all', 
             representations='all', segmentations='all',
             data_dir = DATA_DIR, results_dir = RESULTS_DIR, **kwargs):
    """Evaluate an experiment and store accuracy and retrieval scores in a
    single CSV file that can be used to e.g. generate tables and figures."""
    logging.info('Evaluating experiment...')
    results_dir = os.path.join(results_dir, experiment_name)
    conditions, _ = get_conditions(
        classifiers, genres, subsets, representations, segmentations)
    scores = []
    for condition in conditions:
        segmentation = condition['segmentation']
        output_dir = os.path.join(
            results_dir, condition['classifier'], condition['genre'], condition['subset'])
        test_pred_fn = os.path.join(
            output_dir, condition['representation'], f'{segmentation}-test-pred.txt')
        train_pred_fn = os.path.join(
            output_dir, condition['representation'], f'{segmentation}-train-pred.txt')
        
        condition_scores = get_scores(
            test_pred_fn = test_pred_fn,
            train_pred_fn = train_pred_fn,
            genre=condition['genre'],
            subset=condition['subset'],
            data_dir=data_dir)
        condition_scores.update(condition)
        scores.append(condition_scores)
            
    scores_fn = os.path.join(results_dir, f'{experiment_name}-scores.csv')
    pd.DataFrame(scores).to_csv(scores_fn)
    logging.info(f'> Stored scores to {relpath(scores_fn)}')