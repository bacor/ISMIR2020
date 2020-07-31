# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2020 Bas Cornelissen
# License: MIT
# -----------------------------------------------------------------------------
"""
Segmentations
=============

This module contains various segmentation functions that slice up volpiano 
strings in various ways. Most importantly, the segment_words, segment_syllables
and segment_neumes functions split the notes based on the textual segments.
The Poisson baseline segments the notes randomly, but ensures the segments 
lengths follow a rough Poisson distribution. The k-mer segmentation, finally,
segments a string in successive, non-overlapping substrings of fixed length k.
"""
import re
import numpy as np

def segment_kmers(string: str, k: int = 1) -> list:
    """Segment a string in k-mers: segments of length k. Note that the final
    segment might be shorter than k.

    >>> segment_kmers('abcdefghijklmno', k=2)
    ['ab', 'cd', 'ef', 'gh', 'ij', 'kl', 'mn', 'o']
    >>> segment_kmers('abcdefghijklmno', k=3)
    ['abc', 'def', 'ghi', 'jkl', 'mno']
    >>> segment_kmers('abcdefghijklmno', k=4)
    ['abcd', 'efgh', 'ijkl', 'mno']

    Parameters
    ----------
    string : str
        The string to segment
    k : int, optional
        The length k of the segments, by default 1
    
    Returns
    -------
    list
        A list of segments
    """
    segments = []
    for i in range(0, len(string), k):
        segment = string[i:i+k]
        segments.append(segment)
    return segments

def segment_words(volpiano: str) -> list:
    """Segment a volpiano string in segments corresponding to words.
    Any group of 3 or more dashes is a word boundary.

    >>> segment_words('f--g---h-g')
    ['fg', 'hg']
    >>> segment_words('---f-------g---')
    ['f', 'g']

    Parameters
    ----------
    volpiano : str
        the volpiano string to segment

    Returns
    -------
    list
        A list of word-segments
    """
    # Replace >3 dashes with word boundary
    volpiano = re.sub('-{4,}', '---', volpiano)
    # Word boundary --> space and remove neume/syll boundaries
    volpiano = re.sub('---', ' ', volpiano).replace('-', '').strip()
    words = volpiano.split()
    return words

def segment_syllables(volpiano: str) -> list:
    """Segment a string in segments corresponding to syllables. Any group of 2 
    or more dashes is a syllable boundary.

    >>> segment_syllables('f-g--hg---f------')
    ['fg', 'hg', 'f']

    Parameters
    ----------
    volpiano : str
        The string to segment

    Returns
    -------
    list
        A list of syllable-segments
    """
    volpiano = re.sub('-{3,}', '--', volpiano)
    volpiano = re.sub('--', ' ', volpiano).replace('-', '').strip()
    syllables = volpiano.split()
    return syllables
    
def segment_neumes(volpiano: str) -> list:
    """Segment a string in segments corresponding to neumes. Any group of 1 or 
    more dashes is a neume boundary.

    >>> segment_neumes('fg-h-f---g--')
    ['fg', 'h', 'f', 'g']

    Parameters
    ----------
    volpiano : str
        The volpiano string to segment

    Returns
    -------
    list
        A list of neume-segments
    """
    volpiano = re.sub('-{2,}', '-', volpiano)
    volpiano = re.sub('-', ' ', volpiano).strip()
    neumes = volpiano.split()
    return neumes

def positive_poisson_sample(lam: float) -> int:
    """Return a sample from a shifted Poisson distribution
    where all weight is on the positive integers 1, 2, 3, ...
    So if `x` is a sample from this distribution, then (x-1) is
    Poisson(mu-1)-distributed.
    
    Parameters
    ----------
    lam : float
        The Poisson parameter
    
    Returns
    -------
    int
        A sample from the distribution
    """
    return 1 + np.random.poisson(lam=lam - 1)

def segment_poisson(iterable, lam: float) -> list:
    """Segment an iterable in segments whose length is approximately
    Poisson-distributed. Note that the first and final segment are always
    discarded. Also, the segment lengths cannot be zero, so the
    lengths actually follow a shifted poisson distribution.

    >>> np.random.seed(0)
    >>> segment_poisson('123456789', lam=1)
    ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    >>> segment_poisson('123456789', lam=3)
    ['1234', '567', '89']
    >>> segment_poisson('123456789', lam=6)
    ['123', '456789']
    
    Parameters
    ----------
    iterable : iterable
        The iterable to segment
    lam : float
        The parameter of the poisson
    
    Returns
    -------
    list
        A list of segments
    """
    if lam < 1: raise ValueError('Lambda should be > 1.')
    segments = []
    last_idx = 0
    segment_length = positive_poisson_sample(lam)
    while last_idx + segment_length < len(iterable):
        segment = iterable[last_idx:last_idx+segment_length]
        segments.append(segment)
        last_idx += segment_length
        segment_length = positive_poisson_sample(lam)

    if last_idx < len(iterable):
        segments.append(iterable[last_idx:])
    
    return segments
