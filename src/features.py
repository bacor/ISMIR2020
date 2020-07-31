# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2020 Bas Cornelissen
# License: MIT
# -----------------------------------------------------------------------------
"""
Features
========

This module contains functions that extract basic melodic features from a list
of MIDI pitches, which are used in the baseline mode classification.
"""
import numpy as np
from collections import Counter
from .representation import volpiano_to_midi
from .representation import MIN_MIDI_PITCH, MAX_MIDI_PITCH

def initial(pitches: list) -> int:
    """The initial pitch

    >>> initial([60, 62, 64, 65, 64])
    60

    Parameters
    ----------
    pitches : list
        A list of MIDI pitches 

    Returns
    -------
    int 
        The initial pitch
    """
    return pitches[0]

def final(pitches: list) -> int:
    """The final pitch

    >>> final([60, 62, 64, 65, 64])
    64

    Parameters
    ----------
    pitches : list
        List of MIDI pitches

    Returns
    -------
    int
        The final pitch
    """
    return pitches[-1]

def lowest(pitches: list) -> int:
    """The lowest pitch

    >>> lowest([60, 62, 64, 65])
    60

    Parameters
    ----------
    pitches : list
        list of MIDI pitches

    Returns
    -------
    int
        The lowest pitch
    """

    return min(pitches)

def highest(pitches: list) -> int:
    """The highest pitch

    >>> highest([60, 62, 64, 65])
    65

    Parameters
    ----------
    pitches : list
        List of MIDI pitches

    Returns
    -------
    int
        The highest pitch
    """
    return max(pitches)

def ambitus(pitches: list) -> int:
    """The ambitus or range of a melody in semitones

    >>> ambitus([60, 62, 64, 65, 60])
    5

    Parameters
    ----------
    pitches : list
        List of MIDI pitches

    Returns
    -------
    int
        The ambitus
    """
    return max(pitches) - min(pitches)

def initial_gesture(pitches: list, length: int = 3):
    """The initial gesture of a melody

    >>> initial_gesture([60, 62, 64, 65, 67])
    [60, 62, 64]

    Parameters
    ----------
    pitches : list
        List of MIDI pitches
    length : int, by default 3
        The length of the initial gesture

    Returns
    -------
    list
        A list of the first `length` pitches
    """
    return pitches[:length]

def pitch_profile(pitches: list) -> np.array:
    """Compute a pitch profile from a set of pitches. This is a vector of 
    length 34 containing the relative frequency of the MIDI pitches 53, ..., 86
    in the chant.

    >>> profile = pitch_profile([60, 60, 62, 64, 62])
    >>> len(profile)
    34
    >>> profile[60 - 53]
    0.4
    >>> profile[62 - 53]
    0.4
    >>> profile[64 - 53]
    0.2

    Parameters
    ----------
    pitches : list
        A list of MIDI pitches

    Returns
    -------
    np.array
        The pitch profile
    """
    counts = Counter(pitches)
    profile = [counts[p] for p in range(MIN_MIDI_PITCH, MAX_MIDI_PITCH+1)]
    return np.array(profile) / sum(profile)

def pitch_class_profile(pitches: list) -> np.array:
    """Compute a pitch class profile: a vector of length 12 containing the 
    relative frequency of each pitch class 0, ..., 11 in the list of pitches

    >>> pitch_class_profile([60, 60, 62, 64, 62])
    array([0.4, 0. , 0.4, 0. , 0.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. ])

    Parameters
    ----------
    pitches : list
        A list of MIDI pitches

    Returns
    -------
    np.array
        The pitch class profile
    """
    classes = [p % 12 for p in pitches]
    counts = Counter(classes)
    profile = [counts[p] for p in range(0, 12)]
    return np.array(profile) / sum(profile)

def repetition_profile(pitches: list) -> np.array:
    """Repetition profile describing how often every pitch is repeated in the
    list of pitches.

    If a list of N pitches is passed, there are N-1 possible repetitions.
    The function counts how often each pitch is repeated directly, and
    normalizes this by N-1. This is returned as a vector of length 34 
    corresponding to the 34 pitches supported by Volpiano: the MIDI pitches
    53, ..., 86.

    >>> profile = repetition_profile([60, 61, 62, 63])
    >>> len(profile)
    34
    >>> np.all(profile == 0)
    True
    >>> start, end = 60 - 53, 65-53
    >>> repetition_profile([60, 60, 60])[start:end]
    array([1., 0., 0., 0., 0.])
    >>> repetition_profile([60, 60, 61])[start:end]
    array([0.5, 0. , 0. , 0. , 0. ])
    >>> repetition_profile([60, 60, 60, 61, 61, 62, 60, 60, 60])[start:end]
    array([0.5  , 0.125, 0.   , 0.   , 0.   ])

    Parameters
    ----------
    pitches : list
        List of MIDI pitches

    Returns
    -------
    np.array
        The repetition profile
    """
    repetitions = Counter()
    for prev, cur in zip(pitches[:-1], pitches[1:]):
        if prev == cur:
            repetitions[prev] += 1        
    profile = [repetitions[p] for p in range(MIN_MIDI_PITCH, MAX_MIDI_PITCH+1)]
    return np.array(profile) / (len(pitches) - 1)