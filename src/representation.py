# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2020 Bas Cornelissen
# License: MIT
# -----------------------------------------------------------------------------
"""
Representations
===============

This module contains several converters that turn volpiano strings in interval
or contour representations. These are all string-based, so intervals and 
contours are encoded as strings.
"""
from .volpiano import volpiano_characters

MIN_MIDI_PITCH = 53
MAX_MIDI_PITCH = 86
VOLPIANO_TO_MIDI = {
    "8": 53, # F
    "9": 55, # G
    "a": 57,
    "y": 58, # B flat
    "b": 59,
    "c": 60,
    "d": 62,
    "w": 63, # E flat
    "e": 64,
    "f": 65,
    "g": 67,
    "h": 69,
    "i": 70, # B flat
    "j": 71,
    "k": 72, # C
    "l": 74,
    "x": 75, # E flat
    "m": 76,
    "n": 77,
    "o": 79,
    "p": 81,
    "z": 82, # B flat
    "q": 83, # B
    "r": 84, # C
    "s": 86,
    
    # Liquescents
    "(": 53,
    ")": 55,
    "A": 57,
    "B": 59,
    "C": 60,
    "D": 62,
    "E": 64,
    "F": 65,
    "G": 67,
    "H": 69,
    "J": 71,
    "K": 72, # C
    "L": 74,
    "M": 76,
    "N": 77,
    "O": 79,
    "P": 81,
    "Q": 83,
    "R": 84, # C
    "S": 86, # D
    
    # Naturals
    "Y": 59, # Natural at B
    "W": 64, # Natural at E
    "I": 71, # Natural at B
    "X": 76, # Natural at E
    "Z": 83,
}

# There are no intervals larger than +19 or -19 semitones,
# so this coding should suffice:
INTERVAL_ENCODING = {
    -23: "n",
    -22: "m",
    -21: "l",
    -20: "k",
    -19: "j",
    -18: "i",
    -17: "h",
    -16: "g",
    -15: "f",
    -14: "e",
    -13: "d",
    -12: "c",
    -11: "b",
    -10: "a",
    -9:  "₉",
    -8:  "₈",
    -7:  "₇",
    -6:  "₆",
    -5:  "₅",
    -4:  "₄",
    -3:  "₃",
    -2:  "₂",
    -1:  "₁",
    0:   "-",
    1:   "¹",
    2:   "²",
    3:   "³",
    4:   "⁴",
    5:   "⁵",
    6:   "⁶",
    7:   "⁷",
    8:   "⁸",
    9:   "⁹",
    10:  "A",
    11:  "B",
    12:  "C",
    13:  "D",
    14:  "E",
    15:  "F",
    16:  "G",
    17:  "H",
    18:  "I",
    19:  "J",
    20:  "K",
    21:  "L",
    22:  "M",
    23:  "N",
    None: '.'
}

CONTOUR_ENCODING = {
    None: '.',
    0: '-',
    'up': '⌃',
    'down': '⌄'
}

def volpiano_to_midi(volpiano: str, fill_na: bool = False, 
                     skip_accidentals: bool = False) -> list:
    """Translates volpiano pitches to a list of midi pitches

    All non-note characters are ignored or filled with `None`, if `fill_na=True`
    Unless `skip_accidentals=True`, accidentals are converted to midi pitches
    as well. So an i (flat at the B) becomes 70, a B flat. Or a W (a natural at
    the E) becomes 64 (E).

    >>> volpiano_to_midi('cdefghjk')
    [60, 62, 64, 65, 67, 69, 71, 72]
    >>> volpiano_to_midi('c3d', fill_na=False)
    [60, 62]
    >>> volpiano_to_midi('c3d', fill_na=True)
    [60, None, 62]
    >>> volpiano_to_midi('ij', skip_accidentals=False)
    [70, 71]
    >>> volpiano_to_midi('ij', skip_accidentals=True)
    [71]

    Parameters
    ----------
    volpiano : str
        The volpiano string
    fill_na : bool, optional
        Whether to fill non-notes with None (True), or ignore them (False), by 
        default False
    skip_accidentals : bool, optional
        Whether to skip accidentals. If `skip_accidentals=False`, accidentals 
        are converted to midi pitches. So an `i` (flat at the B) becomes 70, 
        a B flat. Or a W (a natural at the E) becomes 64 (E).
        By default False

    Returns
    -------
    list
        A list of MIDI pitches
    """
    accidentals = volpiano_characters('flats', 'naturals')
    midi = []
    for char in volpiano:
        if skip_accidentals and char in accidentals:
            pass
        elif char in VOLPIANO_TO_MIDI:
            midi.append(VOLPIANO_TO_MIDI[char])
        elif fill_na:
            midi.append(None)
    return midi

def notes_to_intervals(notes: list) -> list:
    """Convert a list of MIDI pitches to a list of intervals. 

    >>> notes_to_intervals([60, 62, 64, 60])
    [2, 2, -4]

    Parameters
    ----------
    notes : list
        The notes, as MIDI pitches

    Returns
    -------
    list
        The intervals
    """
    pairs = zip(notes[:-1], notes[1:])
    get_interval = lambda pair: pair[1] - pair[0]
    intervals = list(map(get_interval, pairs))
    return intervals

def encode_intervals(intervals: list) -> str:
    """Encode an iterable of intervals to its string representation

    >>> encode_intervals([1, 2, 3, 4, 0, -4, -3, -2, -1])
    '¹²³⁴-₄₃₂₁'

    Parameters
    ----------
    intervals : list
        A list of intervals in semitones

    Returns
    -------
    str
        A string encoding of the list of intervals
    """
    encoded_intervals = [INTERVAL_ENCODING[i] for i in intervals]
    return "".join(encoded_intervals)

def encode_contour(intervals):
    """Contour representation of a list of intervals

    >>> encode_contour([None, 2, 0, -2])
    '.⌃-⌄'

    Parameters
    ----------
    intervals : list
        List of intervals (integers)

    Returns
    -------
    str
        The contour
    """
    contour = []
    for interval in intervals:
        if interval is None:
            contour.append(CONTOUR_ENCODING[None])
        elif interval is 0:
            contour.append(CONTOUR_ENCODING[0])
        elif interval > 0:
            contour.append(CONTOUR_ENCODING['up'])
        elif interval < 0:
            contour.append(CONTOUR_ENCODING['down'])      
    return "".join(contour)

def copy_segmentation(source, target, sep=' ', validate=True):
    """Copy the segmentation from a source string to the target string
    
    ```
    >> copy_segmentation('ab cde', '12345')
    '12 345'
    >> copy_segmentation('ab|cd|e', '12345', sep='|')
    '12|34|5'
    >> copy_segmentation('ab cd', '123456')
    ValueError: source and target should have the same number of (non-sep) characters
    ```
    """
    # Test input
    source_chars = source.replace(sep, "")
    if not len(source_chars) == len(target):
        raise ValueError('source and target should have the same number of (non-sep) characters')
    
    # Copy segmentation
    start = 0
    target_units = []
    source_units = source.split(sep)
    for i, source_unit in enumerate(source_units):
        target_unit = target[start:start+len(source_unit)]
        target_units.append(target_unit)
        start += len(source_unit)
    
    # Validate result
    if validate:
        assert "".join(target_units) == target
    return sep.join(target_units)

def interval_representation(volpiano: str, segment: bool = True, 
                            first_interval_empty: bool = True,
                            repeat_first_note: bool = False,
                            sep: str = ' ') -> str:
    """Get interval representation of a volpiano string, keeping the segmentation intact.
    
    >>> interval_representation('ab caa b')
    '.² ¹₃- ²'
    >>> interval_representation('ab|caa|b', sep='|')
    '.²|¹₃-|²'
    >>> interval_representation('ab|caa|b', sep='|', segment=False)
    '.²¹₃-²'
    >>> interval_representation('ab caa b', first_interval_empty=False, segment=False)
    '²¹₃-²'
    >>> interval_representation('ab caa b', repeat_first_note=True)
    '-² ¹₃- ²'

    Parameters
    ----------
    volpiano : str
        The volpiano string
    segment : bool, optional
        Segment the intervals?, by default True
    first_interval_empty : bool, optional
        Include a first empty interval? This is required for the segmentation 
        to work. By default True.
    repeat_first_note : bool, default False
        Whether to repeat the first note.
    sep : str, optional
        The separator used to segment the input volpiano string, by default ' '

    Returns
    -------
    str
        [description]
    """
    notes = volpiano_to_midi(volpiano)
    if repeat_first_note:
        first_interval_empty = False
        notes = [notes[0]] + notes

    intervals = notes_to_intervals(notes)
    if first_interval_empty:
        intervals = [None] + intervals

    encoded = encode_intervals(intervals)
    if segment:
        return copy_segmentation(volpiano, encoded, sep=sep)
    else:
        return encoded

def contour_representation(volpiano: str, segment: bool = True, 
                           first_interval_empty: bool = True,
                           repeat_first_note: bool = False,
                           sep: str =' ') -> str:
    """Get a contour representation of a volpiano string

    >>> contour_representation('ab caa b')
    '.⌃ ⌃⌄- ⌃'
    >>> contour_representation('ab|caa|b', sep='|')
    '.⌃|⌃⌄-|⌃'
    >>> contour_representation('ab|caa|b', sep='|', segment=False)
    '.⌃⌃⌄-⌃'
    >>> contour_representation('ab caa b', first_interval_empty=False, segment=False)
    '⌃⌃⌄-⌃'
    >>> contour_representation('ab caa b', repeat_first_note=True)
    '-⌃ ⌃⌄- ⌃'

    Parameters
    ----------
    volpiano : str
        The volpiano string
    segment : bool, optional
        Whether to segment the contour, by default True
    first_interval_empty : bool, optional
        Include a first empty interval? This is required for the segmentation 
        to work. By default True.
    repeat_first_note : bool, default False
        Whether to repeat the first note.
    sep : str, optional
        The segment-separator in the input string, by default ' '

    Returns
    -------
    str
        The contour
    """
    notes = volpiano_to_midi(volpiano)
    if repeat_first_note:
        first_interval_empty = False
        notes = [notes[0]] + notes

    intervals = notes_to_intervals(notes)
    if first_interval_empty:
        intervals = [None] + intervals
    
    contour = encode_contour(intervals)
    if segment:
        return copy_segmentation(volpiano, contour, sep=sep)
    else:
        return contour
