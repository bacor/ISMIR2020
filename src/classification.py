import os 
import time
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from .helpers import save_model
from .helpers import md5checksum
from .helpers import relpath
from .helpers import param_str
from .helpers import read_predictions
from .helpers import DATA_DIR, ROOT_DIR

def tune_model(model, data, targets, param_grid, n_splits, n_iter):
    """"""
    rs = np.random.randint(100)
    logging.info(f'> Tuning classfier using stratified {n_splits}-fold CV and random search')
    logging.info(f'  Tune using randomized search (n_iter={n_iter}, random_state={rs})')
    logging.info(f'  Tuned params: {list(param_grid.keys())}')
    t0 = time.time()

    # Tune!
    tuner = RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_grid, 
        scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
        cv=StratifiedKFold(n_splits=n_splits), 
        refit=False,
        n_iter=n_iter,
        n_jobs=-1,
        return_train_score=True,
        random_state=rs)
        
    tuner.fit(data, targets)
    cv_results = pd.DataFrame(tuner.cv_results_).sort_values('rank_test_accuracy')
    cv_results.index.name = 'id'
        
    # Log The best results
    best = cv_results.iloc[0, :]
    logging.info(f'  Best params: {best.params}')
    logging.info( '  Mean cross validation statistics for best params:')
    
    train_acc = best['mean_train_accuracy']
    test_acc = best['mean_test_accuracy']
    logging.info(f'  - Train/test accuracy: {train_acc:.2%}%/{test_acc:.2%}% (rank {best.rank_test_accuracy})')

    train_prec = best['mean_train_precision_weighted']
    test_prec = best['mean_test_precision_weighted']
    logging.info(f'  - Train/test precision: {train_prec:.2%}%/{test_prec:.2%}% (rank {best.rank_test_precision_weighted})')

    train_rec = best['mean_train_recall_weighted']
    test_rec = best['mean_test_recall_weighted']
    logging.info(f'  - Train/test recall: {train_rec:.2%}%/{test_rec:.2%}% (rank {best.rank_test_recall_weighted})')

    train_f1 = best['mean_train_f1_weighted']
    test_f1 = best['mean_test_f1_weighted']
    logging.info(f'  - Train/test f1: {train_f1:.2%}%/{test_f1:.2%}% (rank {best.rank_test_f1_weighted})')

    # Update model parameters
    logging.info('  Setting model parameters to highest-accuracy params')
    model.set_params(**best.params)

    duration = time.time() - t0
    logging.info(f'  Tuning done in {duration:.2f}s')
    return cv_results

def train_model(model, 
                train_data, train_targets, test_data, test_targets, 
                param_grid, n_splits, n_iter,
                basepath):
    """"""
    # Tune the model
    cv_results = tune_model(
        model=model, 
        data=train_data, 
        targets=train_targets, 
        param_grid=param_grid, 
        n_splits=n_splits, 
        n_iter=n_iter)
    
    # Train the model
    t0 = time.time()
    model.fit(train_data, train_targets)
    duration = time.time() - t0
    logging.info(f'> Training done in {duration:.2f}s')

    # Serialize trained model
    model_fn = f'{basepath}.model'
    save_model(model, model_fn)
    logging.info(f'> Serialized model to {relpath(model_fn)}')
    logging.info(f'  md5 checksum: {md5checksum(model_fn)}')

    # Store tuning results
    tuning_fn = f'{basepath}-tuning.csv'
    cv_results.to_csv(tuning_fn)
    logging.info(f'> Stored tuning results to {relpath(tuning_fn)}')
    logging.info(f'  md5 checksum: {md5checksum(tuning_fn)}')

    # Train predictions and accuracy
    train_pred_fn = f'{basepath}-train-pred.txt'
    train_pred = model.predict(train_data)
    train_pred.tofile(train_pred_fn, sep='\n')
    logging.info(f'> Stored train predictions to {relpath(train_pred_fn)}')
    logging.info(f'  md5 checksum: {md5checksum(train_pred_fn)}')
    train_acc = accuracy_score(train_targets, train_pred)
    logging.info(f'  Train accuracy: {train_acc:.2%}')

    # Test predictions and accuracy
    test_pred_fn = f'{basepath}-test-pred.txt'
    test_pred = model.predict(test_data)
    test_pred.tofile(test_pred_fn, sep='\n')
    logging.info(f'> Stored test predictions to {relpath(test_pred_fn)}')
    logging.info(f'  md5 checksum: {md5checksum(test_pred_fn)}')
    test_acc = accuracy_score(test_targets, test_pred)
    logging.info(f'  Test accuracy: {test_acc:.2%}')

    logging.info(f'> Done.')
    logging.info('*' * 60)

def get_scores(train_pred_fn, test_pred_fn, genre, subset, data_dir=DATA_DIR, 
               **kwargs):
    """Evaluate model predictions and return evaluation scores as a dict"""
    train_fn = os.path.join(data_dir, genre, subset, f'train-chants.csv')
    test_fn = os.path.join(data_dir, genre, subset, f'test-chants.csv')
    train_targets = pd.read_csv(train_fn)['mode']
    test_targets = pd.read_csv(test_fn)['mode']
    train_predictions = read_predictions(train_pred_fn)
    test_predictions = read_predictions(test_pred_fn)
    train_report = classification_report(
        train_targets, train_predictions, output_dict=True)
    test_report = classification_report(
        test_targets, test_predictions, output_dict=True)

    return dict(
        genre=genre,
        subset=subset,
        train_accuracy = accuracy_score(train_targets, train_predictions),
        test_accuracy = accuracy_score(test_targets, test_predictions),
        train_weighted_f1 = train_report['weighted avg']['f1-score'],
        train_weighted_precision = train_report['weighted avg']['precision'],
        train_weighted_recall = train_report['weighted avg']['recall'],
        test_weighted_f1 = test_report['weighted avg']['f1-score'],
        test_weighted_precision = test_report['weighted avg']['precision'],
        test_weighted_recall = test_report['weighted avg']['recall'],
    )