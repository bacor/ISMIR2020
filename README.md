Mode Classification and Natual Units in Plainchant
==================================================

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