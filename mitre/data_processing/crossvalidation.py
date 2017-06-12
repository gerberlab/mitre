""" Utilities for cross-validation. """

import itertools
import numpy as np
from .transforms import select_variables, select_subjects
import transforms

def stratified_k_folds(dataset,folds=5):
    """ Split rules.Dataset object into pieces for k-fold CV.

    The ratio of positive to negative outcomes in each fold will
    be (approximately) equal.

    Returns: [(train_1, test_1), ..., (train_k, test_k)]: 
    list of pairs of rules.Dataset objects. 

    """
    N_subjects = dataset.n_subjects
    all_indices = np.arange(N_subjects)
    unused_test_indices = set(all_indices) 
    # dataset.y is generally boolean, but account for the case
    # where it may be a vector of 0/1
    control_indices = all_indices[np.logical_not(dataset.y)]
    case_indices = all_indices[dataset.y.astype(bool)]
    if len(case_indices)==0 or len(control_indices)==0:
        raise ValueError('Stratification failure (uniform outcome?)')
    np.random.shuffle(control_indices)
    np.random.shuffle(case_indices)
    results = []
    # Deal the case and control indices out one by one to the test
    # sets for each fold, so that no fold has more than one more (or
    # less) case or control than any other
    test_indices_by_fold = [[] for i in xrange(folds)]
#    for indices_to_deal in (control_indices, case_indices):
    for index, fold in zip(list(case_indices) + list(control_indices), 
                           itertools.cycle(test_indices_by_fold)):
            fold.append(index)

    for test_indices in test_indices_by_fold:
        test = transforms.select_subjects(dataset, test_indices)
        train = transforms.select_subjects(dataset, test_indices,
                                           invert=True)
        results.append((train, test))
    return results


def leave_one_out_folds(dataset):
    """ Split rules.Dataset object into pieces for leave-one-out CV.

    Returns: [(train_1, test_1), ..., (train_k, test_k)]: 
    list of pairs of rules.Dataset objects. 

    """
    N_subjects = dataset.n_subjects
    test_indices_by_fold = [[i] for i in xrange(N_subjects)]
    results = []
    for test_indices in test_indices_by_fold:
        test = transforms.select_subjects(dataset, test_indices)
        train = transforms.select_subjects(dataset, test_indices,
                                           invert=True)
        results.append((train, test))
    return results


def debug_leave_one_out_folds(dataset):
    """ Split rules.Dataset object into pieces for leave-one-out CV.

    This method produces only two folds.

    Returns: [(train_1, test_1), ..., (train_k, test_k)]: 
    list of pairs of rules.Dataset objects. 

    """
    N_subjects = dataset.n_subjects
    test_indices_by_fold = [[i] for i in xrange(N_subjects)]
    results = []
    for test_indices in test_indices_by_fold[:2]:
        test = transforms.select_subjects(dataset, test_indices)
        train = transforms.select_subjects(dataset, test_indices,
                                           invert=True)
        results.append((train, test))
    return results
