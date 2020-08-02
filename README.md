Mode Classification and Natural Units in Plainchant
==================================================

This repository contains the data and code for the paper 'Mode Classification 
and Natural Units in Plainchant' accepted at the [International Society
for Music Information Retrieval (ISMIR)
Conference 2020](https://ismir.github.io/ISMIR2020/).

----

<img src="figures/teaser/teaser.jpg?raw=true" width="800" 
    title="Three approaches to mode classification in plainchant compared">

**Summary.**
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

**Reference.**
Bas Cornelissen, Willem Zuidema, and John Ashley Burgoyne, 
“Mode classification and natural units in plainchant”, in 
*Proc. of the 21st Int. Society for Music Information Retrieval Conf.*, 
Montréal, Canada, 2020

---

&nbsp;

&nbsp;
Contents
--------

1. [Repository structure](#repository-structure)
2. [Python setup](#python-setup) 
3. [Generating the data](#generating-the-data) 
4. [Running the experiments](#running-the-experiments) 
5. [Analyzing the results](#analyzing-the-results)

Repository structure 
--------------------

- **`cantuscorpus/`** The corpus used in this study is not included in the 
repository, but can be downloaded here: 
[CantusCorpus v0.2](https://github.com/bacor/cantuscorpus/releases/tag/v0.2). 
Just remove the `-v0.2` from the directory name, and place it in the root of the
repository,
- **`data/`** Contains the  data used in the experiments. We report results for five 
independent runs of whole study, for which we generated five datasets with 
different train/test splits, named `run-0` to `run-4`. Only the data for the
first run is included in the repository, but the other datasets can be 
reproduced as the generation is deterministic. Data per run is further structured 
as follows:
    - `data/run-[i]/[genre]/[subset]/[split]-chants.csv`: a chants file with 
    details about the chant, including the volpiano and mode.
    - `data/run-[i]/[genre]/[subset]/[split]-features.csv`: a table with all
    features used by both the classical and the profile-based approach.
    - `data/run-[i]/[genre]/[subset]/[split]-representation-[representation].csv`
    A table containing chants in the given representation, segmented in many
    different ways: besides the natural segmentations, 1–16 grams (or *k-mers*)
    and three random baselines.
where `genre` can be `antiphon` or `responsory`; `subset` can be `full` (all 
chants) or `subset`, meaning only the subset without melody variants; `split` 
can be `train` or `test` and the `representation` can be `pitch`, 
`interval-dependent`, `interval-independent`, `contour-dependent` and
`contour-independent`.
- **`demo-data/`** This folder has the same structure as `data`, but is generated using the
`chant-demo-sample.csv` table from CantusCorpus. This demo data is useful
during development.
- **`experiments/`** Every experiment has a number of parameters, like the type of
model, the number of cross-validation splits, but also what directory to load
the data from. To record which parameters where used to produce which results, 
we specify the experiment parameters in YAML files in the `experiments` folder.
- **`figures/`**: all figures made in this study. Most plots are generated using
the notebooks in `notebooks/`, and then finalized in Affinity Designer (those
files are not included). The directory also contains many figures that are not
in the paper or the supplements, such heatmaps with other evaluation metrics.
- **`notebooks/`** Contains the Jupyter notebooks used to generate the figures, 
or to do other analyses.
- **`src/`** Contains all code used to generate the datasets, run the experiments and 
compute tf–idf vector embeddings. All Python files are documented.
- **`tests/`** Contains some unittests for some of the code in `src/`.


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

Generating the data
-------------------

```bash
# Generate the complete dataset
python -m src.generate_data --seed=0

# Generate a demo dataset
python -m src.generate_data --what=demo --seed=0
```

Here `seed` is a number used to set the random seed. This is used to generate
five different datasets used in five independent runs (with seeds 0, ..., 5).

Running the experiments
-----------------------

```bash
python cli.py run experiments/profile-demo.yml
python cli.py run experiments/profile-run-0.yml
python cli.py run experiments/profile-run-1.yml
# ...
```

Analyzing the results
---------------------

All plots are made in the Jupyter notebooks in `notebooks/`. However, the
low-dimensional embeddings of the tf-idf vectors are computed in 
`src/tfidf_visualization.py`; there's no cli for this, but tweaking the script
is straigtforward.


License
-------

All code is released under an MIT licence. The figures are released under a
CC-BY 4.0 license.