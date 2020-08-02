Mode Classification and Natural Units in Plainchant
==================================================

This repository contains all code for the paper 'Mode Classification and Natural 
Units in Plainchant' presented at [ISMIR 2020](https://ismir.github.io/ISMIR2020/).

----

<img src="figures/teaser/teaser.jpg?raw=true" width="800" 
    title="Three approaches to mode classification in plainchant compared">

**Main findings.**
Our study compares three approaches to classifying mode in medieval plainchant
from the Cantus database. The eight modes are the central tonalities around 
which the repertoire is organized. The *classical approach* classifies melodies
to modes by looking at the final note and the range of the melody, and possibly
other features. The *profile approach* looks at pitch (class) profiles, just like
is often done when determining musical key in later Western music. Finally, the
*distributional approach* represents melodies as frequency vectors counting, 
with some weighing, how often certain melodic fragments (units) occur in the melody.
We try all sorts of units, including three *natural units*: notes that form 
so-called neumes, syllables or words. Overall, the distributional approach 
works best, and of all units that can be used with it, natural units work best.
In fact, they work surprisingly well even if we throw away the actual pitches 
and use  only the intervals in a melody, or worse still, if we use only whether 
the melody goes up or down (its contour). Could this mean that, just like a 
sentence is made by stringing together words, a chant melody is made together
by concatenating small musical motifs?

---


Repository structure 
--------------------

- `cantuscorpus`: the corpus used in this study is not included in the 
repository, but can be downloaded here: 
[CantusCorpus v0.2](https://github.com/bacor/cantuscorpus/releases/tag/v0.2). 
Just remove the `-v0.2` from the directory name, and place it in the root of the
repository,
- `data/`: the data used in all experiments. This directory
    - `data/[genre]/full`: all antiphon data
    - `data/[genre]/subset`: a subset where we made sure that every chant had a 
    unique (cantus_id, mode) combination. This effectively means there are no 
    (or in any case fewer) melody variants in the corpus.
- `demo-data/`: has the same structure as `data`, but is generated using the
`chant-demo-sample.csv` table from CantusCorpus. This demo data is useful
during development.
- `experiments`: a directory with yml files containing experiment settings
- `src/`:


Python setup
------------

You can find the Python version used in `.python-version` and all dependencies 
are listed in `requirements.txt`. If you use `pyenv` and `venv` to manage 
python versions and virtual environments, do the following:

```bash
# Install the right python version
pyenv install | cat .python-version

# Create a virtual environment
python -m venv env

# Activate the environment
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```


Then you can start Jupyter Lab to run the notebooks:

```bash
$ jupyter lab
```

Generating the data
-------------------

To generate the data

```bash
# Generate the complete dataset
python -m src.generate_data

# Generate a demo dataset
python -m src.generate_data --what=demo
```