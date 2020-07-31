# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2020 Bas Cornelissen
# License: MIT
# -----------------------------------------------------------------------------
"""
Chant filters
=============

This module contains various functions that help you filtering out certain 
chants from a chants dataframe. All filters are decorated to make sure that
they log the effects of filtering: how many chants were excluded, how many 
remain? 

>>> import pandas as pd
>>> chants = pd.read_csv('cantuscorpus/csv/chant-demo-sample.csv')
>>> df = filter_chants_without_volpiano(chants)
Filter Chants Without Volpiano:
Exclude all chants with an empty volpiano field
 > 0.00% removed (0 out of 2000; 2000 remain)
>>> df = filter_chants_by_genre(chants, include=['genre_a'])
Filter Chants By Genre:
Include only chants with a certain genre
 * include=['genre_a']
 > 58.15% removed (1163 out of 2000; 837 remain)
>>> df = filter_chants_without_simple_mode(chants, include_transposed=False)
Filter Chants Without Simple Mode:
Include only chants with simple modes: 1-8, not transposed
 * include_transposed=False
 > 24.95% removed (499 out of 2000; 1501 remain)

"""
from typing import Callable, Dict
from .volpiano import clean_volpiano

# Filter helpers
# --------------

def log_filter_header(func: Callable, kwargs: Dict, logger: Callable = print):
    """Logs the name, docstring and optional arguments of a filter.

    Parameters
    ----------
    func : callable
        The filter function
    kwargs : dict
        The keyword arguments passed to the filter
    logger : callable, optional
        The logger function, by default print
    """
    logger(func.__name__.title().replace('_', ' ') + ':')
    if func.__doc__:
        logger(func.__doc__)
    for key, value in kwargs.items():
        if not key is 'logger':
            logger(f' * {key}={value}')

def log_filter_results(before: int, after: int, logger: Callable = print):
    """Log the results of the filtering: how many entries were removed
    and how many remain?

    Parameters
    ----------
    before : int
        The size of the dataframe before filtering
    after : int
        The size of the dataframe after filtering
    logger : Callable, optional
        The logger function, by default print
    """
    removed = before - after
    perc_removed = removed / before
    logger(f' > {perc_removed:.2%} removed ({removed} out of {before}; '
           f'{after} remain)')

def log_filter(func: Callable):
    """Decorator that automatically logs the results of filtering. The filter 
    function being decorated should have the form::
    
        @log_filter
        def my_filter(df, my_arg=1, logger=None):
            # ...
            return df
    
    The argument `logger=None` is required. Setting it to `False` 
    disables logging. You can also pass a logger function (default: print)

    Parameters
    ----------
    func : Callable
        The filter function
    
    Returns
    -------
    Callable
        The wrapper
    """
    def func_wrapper(df, **kwargs):
        filtered_df = func(df, **kwargs)
        logger = kwargs.get('logger', print)
        if not logger is False:
            log_filter_header(func, kwargs, logger=logger)
            log_filter_results(len(df), len(filtered_df), logger=logger)
        return filtered_df
    return func_wrapper

# Filters
# -------

@log_filter
def filter_chants_without_volpiano(chants, logger=None):
    """Exclude all chants with an empty volpiano field"""
    has_volpiano = chants.volpiano.isnull() == False
    return chants[has_volpiano]

@log_filter
def filter_chants_without_notes(chants, logger=None):
    """Exclude all chants without notes"""
    notes_pattern = r'[89abcdefghjklmnopqrs\(\)ABCDEFGHJKLMNOPQRS]+'
    contains_notes = chants.volpiano.str.contains(notes_pattern) == True
    return chants[contains_notes]

@log_filter
def filter_chants_with_nonvolpiano_chars(chants, logger=None):
    """Exclude all chants with non-volpiano characters"""
    volpiano_chars = (
        r'3456712\(\)'
        r'ABCDEFGHJKLMNOPQRSIWXYZ89'
        r'abcdefghjklmnopqrsiwxyz'
        r'\.\,\-\[\]\{\¶')
    pattern = f'^[{volpiano_chars}]*$'
    contains_no_other_chars = chants.volpiano.str.match(pattern) == True
    return chants[contains_no_other_chars]

@log_filter
def filter_chants_with_F_clef(chants, logger=None):
    """Exclude chants that contain an F clef"""
    contains_F_clef = chants.volpiano.str.contains('2') == True
    return chants[contains_F_clef == False]

@log_filter
def filter_chants_not_starting_with_G_clef(chants, logger=None):
    """Exclude chants that do not start with a G clef"""
    starts_with_G_clef = chants.volpiano.str.startswith('1') == True
    return chants[starts_with_G_clef]

@log_filter
def filter_chants_with_missing_pitches(chants, logger=None):
    """Filter chants with missing pitches: containing the substring 6------6"""
    has_no_missing_pitches = chants.volpiano.str.contains('6------6') == False
    return chants[has_no_missing_pitches]

@log_filter
def filter_chants_by_genre(chants, include=[], exclude=[], logger=None):
    """Include only chants with a certain genre"""
    genres = chants['genre_id'].unique().tolist()
    if len(include) == 0:
        include = [genre for genre in genres if genre not in exclude]
    has_right_genre = chants['genre_id'].isin(include)
    return chants[has_right_genre]

@log_filter
def filter_chants_without_simple_mode(chants, include_transposed=True, 
                                      logger=None):
    """Include only chants with simple modes: 1-8, not transposed"""
    pattern = '^[1-8]T?$' if include_transposed else '^[1-8]$'
    has_mode = chants['mode'].str.match(pattern) == True
    return chants[has_mode]

@log_filter
def filter_chants_without_full_text(chants, logger=None):
    """Filter chants without full text"""
    has_full_text = chants.full_text.isna() == False
    return chants[has_full_text]

@log_filter
def filter_chants_where_incipit_is_full_text(chants, logger=None):
    """Filter chants whose incipit is identical to the full text"""
    incipit_neq_full_text = chants.full_text != chants.incipit
    return chants[incipit_neq_full_text]

@log_filter
def filter_chants_with_duplicated_notes(chants, logger=None):
    """Filter duplicate chants: whose notes occur multiple times"""
    is_duplicated = chants.volpiano.map(clean_volpiano).duplicated()
    return chants[is_duplicated == False]

@log_filter
def filter_chants_without_word_boundary(chants, logger=None):
    """Only include chants with '---' in their volpiano"""
    constains_word_boundary = chants.volpiano.str.contains('---')
    return chants[constains_word_boundary]

@log_filter
def sample_one_chant_per_mode_and_cantus_id(chants, random_state=0, 
    logger=None):
    """For every cantus_id, sample one chant of each mode."""
    # This results in a  set of unique (cantus_id, mode) pairs. So if there are
    # 4 chants of mode 7 and 2 of mode 3, the function returns two ids: one of 
    # a mode-7 chant, and one of a mode-3 chant.
    ids = []
    seed = random_state
    unique_cantus_ids = chants['cantus_id'].unique()
    for cantus_id in unique_cantus_ids:
        subset = chants.query(f'cantus_id=="{cantus_id}"')
        modes = subset['mode'].unique()
        for mode in modes:
            chant = subset.query(f'mode=="{mode}"').sample(random_state=seed)
            ids.append(chant.index[0])
            seed += 1
    return chants.loc[ids, :]