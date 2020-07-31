# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2020 Bas Cornelissen
# -----------------------------------------------------------------------------
"""
Run an experiment
=================

This script runs the experiment specified by a yaml file. All experiments are 
specified in the `experiments` directory. Usage:

    $ python cli.py run experiments/traditional-demo.yml

See `src/*_experiment.py` for the details of each of the three experiment 
types.
"""
import argparse
import yaml

def command_line_interface():
    # Set up the CLI
    parser = argparse.ArgumentParser(
        description='Run one of the experiments based on a YAML file with'
                    'the preciese settings for the experiment.')
    parser.add_argument(
        'action', type=str, help='What to do? run or evaluate')
    parser.add_argument(
        'settings', type=str, 
        help='YAML file with the settings of the experiment')
    args = parser.parse_args()
    
    # Argument 1: action
    if args.action not in ['run', 'evaluate']:
        raise ValueError('The action should be eiter "run" or "evaluate"')

    # Argument 2: settings
    settings = yaml.safe_load(open(args.settings, 'r'))
    if not 'experiment_type' in settings:
        raise ValueError('You have to specify the experiment_type in the YAML'
            'file. This can be one of "traditional", "profile", and "tf_idf".')
    
    # Load the right module
    if settings['experiment_type'] == 'traditional':
        import src.traditional_experiment as module
    elif settings['experiment_type'] == 'profile':
        import src.profile_experiment as module
    elif settings['experiment_type'] == 'tfidf':
        import src.tfidf_experiment as module

    # Fix random state
    if 'random_state' in settings:
        import numpy as np
        np.random.seed(settings['random_state'])
        del settings['random_state']
    
    # Run!
    del settings['experiment_type']
    if args.action == 'run':
        module.run(**settings)
    if args.action == 'run' or args.action == 'evaluate':
        module.evaluate(**settings)

if __name__ == '__main__':
    command_line_interface()