# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2019 Bas Cornelissen
# License: MIT
# -----------------------------------------------------------------------------
"""
Volpiano helpers
================

Some helpers for working with Volpiano strings.
"""
import re

def expand_accidentals(volpiano, omit_notes=False, barlines='3456'):
    """Expand all accidentals in a volpiano string by adding the accidental
    to all other notes in the scope.
    
    In CANTUS transcriptions, flats are added only once, directly in front
    of the B. This function adds flat signs before all successive Bs until
    the next natural sign, or the next barline. Note that all natural signs
    are removed.

    This function does not assume that when a note is flattened, the notes an
    octave higher and lower are also flattened. (So it assumes that after a 
    central b flat `ij`, a lower b-flat `yb` indeed has an accidental)
 
    Flat/Nat. Note  Description
    --------------------------
    i/I       j     Central b flat
    y/Y       b     Low b flat
    z/Z       q     High b flat
    w/W       e     Low e flat
    x/X       m     High e flat
    
    >>> expand_accidentals('ijjj')
    'ijijij'
    >>> expand_accidentals('ijjj', omit_notes=True)
    'iii'
    >>> expand_accidentals('ij3j', omit_notes=True)
    'i3j'
    >>> expand_accidentals('ijjIjj', omit_notes=True)
    'iijj'
    >>> expand_accidentals('ijjIjj', omit_notes=False)
    'ijijjj'

    Parameters
    ----------
    volpiano : str
        The volpiano string
    omit_notes : bool, optional
        Whether to omit notes. If True, the function only returns the 
        accidentals, not the actual notes. In this case accidentals function
        as notes. By default False
    barlines : str, optional
        Volpiano characters that are barlines and therefore end the scope of
        all accidentals, by default '3456'

    Returns
    -------
    str
        A volpiano string with all flats added
    """
    in_scope = { 'i': False, 'y': False, 'z': False, 'w': False, 'x': False}
    output = ''
    for char in volpiano:
        # If the character is a flat, enter its scope
        if char in 'iyzwx': 
            in_scope[char] = True

        # If a natural, exit the corresponding flats scope
        elif char in 'IYZWX': 
            in_scope[char.lower()] = False
        
        # Reset scope on barlines (single, double, bold or middle)
        elif char in barlines:
            output += char
            for key in in_scope.keys():
                in_scope[key] = False

        # Central b flat?
        elif in_scope['i'] and char == 'j':
            output += 'i' if omit_notes else 'ij'

        # Low b flat?
        elif in_scope['y'] and char == 'b':
            output += 'y' if omit_notes else 'yb'
        
        # High b flat?
        elif in_scope['z'] and char == 'q':
            output += 'z' if omit_notes else 'zq'

        # Low e flat?
        elif in_scope['w'] and char == 'e':
            output += 'w' if omit_notes else 'we'

        # High e flat?
        elif in_scope['x'] and char == 'm':
            output += 'x' if omit_notes else 'xm'

        # Another note
        else:
            output += char
    return output

def clean_volpiano(volpiano, allowed_chars=None, keep_boundaries=False, 
    neume_boundary=' ', syllable_boundary=' ', word_boundary=' ',
    keep_bars=False, allowed_bars='345', bar='|'):
    """Extracts only the allowed characters (and optionally boundaries) from a 
    volpiano string.

    By default, the allowed characters are only notes and accidentals. The 
    cleaning then amounts to removing clefs, bars, etc. The function can retain 
    boundaries, if `add_boundaries=True`. Neume, syllable and word boundaries 
    are then replaced by special boundary markers (`neume_boundary`, 
    `syllable_boundary` and `word_boundary`).

    >>> volpiano = '1---fg---h--ij-h-3-f-4'
    >>> clean_volpiano(volpiano)
    'fghijhf'
    >>> clean_volpiano(volpiano, keep_boundaries=True)
    ' fg h ij h f '
    >>> clean_volpiano(volpiano, keep_bars=True)
    'fghijh|f|'
    >>> bounds = dict(neume_boundary='.', syllable_boundary='-', word_boundary='$')
    >>> clean_volpiano(volpiano, keep_boundaries=True, **bounds)
    '$fg$h-ij.h-f.'
    >>> clean_volpiano(volpiano, keep_boundaries=True, keep_bars=True, **bounds)
    '$fg$h-ij.h|-f|.'

    Parameters
    ----------
    volpiano : str
        The volpiano string
    allowed_chars : str, optional
        A string with all allowed characters. By default this is None, which 
        corresponds to notes, liquescents, flats and naturals
    keep_boundaries : bool, optional
        If true, keeps the boundaries, by default False
    neume_boundary : str, optional
        Neume boundary marker, by default ' '
    syllable_boundary : str, optional
        Syllable boundary markers, by default ' '
    word_boundary : str, optional
        Word boundary marker, by default ' '
    keep_bars : bool, optional
        Whether to keep barlines, by default False
    allowed_bars : str, optional
        Barlines, by default single, double and bold barlines ('345')
    bar : str, optional
        Bar marker, by default '|'

    Returns
    -------
    str
        A clean volpiano string
    """
    if not allowed_chars:
        allowed_chars = volpiano_characters('liquescents', 'notes', 'flats', 'naturals')    

    if keep_boundaries:
        # Remove dashes from the allowed characters
        allowed_chars = ''.join(c for c in allowed_chars if c not in '-')

    output = ''
    num_spaces = 0
    boundaries = { 1: neume_boundary, 2: syllable_boundary, 3: word_boundary }
    for char in volpiano:
        if char in allowed_chars:
            if num_spaces > 0:
                output += boundaries[num_spaces]
                num_spaces = 0
            output += char

        elif keep_boundaries and char == '-':
            num_spaces += 1
            if num_spaces == 3:
                output += word_boundary
                num_spaces = 0

        elif keep_bars and char in allowed_bars:
            output += bar
    
    # Handle spaces at the end
    if num_spaces > 0:
        output += boundaries[num_spaces]
        
    return output

def volpiano_characters(*groups):
    """Returns accepted Volpiano characters

    The characters are organized in several groups: bars, clefs, liquescents,
    naturals, notes, flats, spaces and others. You can pass these group
    names as optional arguments to return only those subsets:

    >>> volpiano_characters()
    '3456712()ABCDEFGHJKLMNOPQRSIWXYZ89abcdefghjklmnopqrsiwxyz.,-[]{¶'
    >>> volpiano_characters('naturals', 'flats')
    'IWXYZiwxyz'

    Parameters
    ----------
    *groups : [str]
        The groups to include: 'bars', 'clefs', 'liquescents', 'naturals', 
        'notes', 'flats', 'spaces' or 'others'

    Returns
    -------
    str
       A string with accepted Volpiano characters
    """
    symbols = {
        'bars': '34567',
        'clefs': '12',
        'liquescents': '()ABCDEFGHJKLMNOPQRS',
        'naturals': 'IWXYZ',
        'notes': '89abcdefghjklmnopqrs',
        'flats': 'iwxyz',
        'spaces': '.,-',
        'others': "[]{¶",
    }
    if not groups:
        groups = symbols.keys()
    return "".join((symbols[key] for key in groups))

def contains_notes(volpiano, accidentals_are_notes=True):
    """Tests whether a volpiano string contains notes, including liquescents.
    By default, accidentals are also treated as notes, but you can disable
    this using the `accidentals_are_notes` parameter.

    >>> contains_notes('1---6')
    False
    >>> contains_notes('fg')
    True
    >>> contains_notes('FG')
    True
    >>> contains_notes('i', accidentals_are_notes=True)
    True
    >>> contains_notes('i', accidentals_are_notes=False)
    False

    Parameters
    ----------
    volpiano : str
        The volpiano string to test
    accidentals_are_notes : bool
        Whether to treat accidentals as notes. By default True.

    Returns
    -------
    bool
        True if the volpiano string contains notes
    """
    groups = ['notes', 'liquescents']
    if accidentals_are_notes: 
        groups.extend(['flats', 'naturals'])
    expr = f'[{volpiano_characters(*groups)}]+'
    return bool(re.search(expr, volpiano))

def has_no_notes(volpiano):
    return contains_notes(volpiano) == False

def split_string(mystring, sep, keep_sep=True):
    """Splits a string, with an option for keeping the separator in

    >>> split_string('this-is-a-test', '-')
    ['this-', 'is-', 'a-', 'test']
    >>> split_string('this-is-a-test', '-', keep_sep=False)
    ['this', 'is', 'a', 'test']

    Parameters
    ----------
    mystring : str
        The string to split
    sep : str
        The separator to split at
    keep_sep : bool, optional
        Whether to keep the separator, by default True

    Returns
    -------
    list
        The parts of the string
    """
    if keep_sep == False:
        keep_sep = ''
    elif keep_sep == True:
        keep_sep = sep
    items = mystring.split(sep)
    for i in range(len(items) - 1):
        items[i] += keep_sep
    return items

def split_volpiano(volpiano, sep, keep_sep=True):
    """Split a Volpiano string, while ignoring the final dashes
    
    >>> split_volpiano('f-g-h--', sep='-')
    ['f-', 'g-', 'h--']
    >>> split_volpiano('f-g-h--', sep='-', keep_sep=False)
    ['f', 'g', 'h--']

    Parameters
    ----------
    volpiano : str
        The string to split
    sep : str
        The separator to split at
    keep_sep : bool, optional
        Whether to keep the separators. Note that the final dashes are always
        included in the last itme. By default True

    Returns
    -------
    list
        The parts
    """
    
    # Find number of final dashes
    num_final_dashes = 0
    try:
        while volpiano[-(num_final_dashes + 1)] == '-':
            num_final_dashes += 1
    except IndexError:
        pass
    
    # Split without the final dashes; then append them to the last item
    if num_final_dashes > 0:
        volpiano = volpiano[:-num_final_dashes]
    items = split_string(volpiano, sep, keep_sep=keep_sep)
    items[-1] += '-' * num_final_dashes
    return items
