""" 
Generate a qualitatively apt test data set for comparing methods.

"""
from rules import linear_test_data
import numpy as np
import random

def demo_data(N_subjects, N_otus, 
              otu_tree, N_timepoints, N_intervals, zero_value):
    """
    
    Calls linear_test_data to generate data with the indicated parameters,
    then makes each leaf node of the tree mutually exclusive with 
    its immediate sister node by (for each subject) picking one of each
    pair at random to include, overwriting the data for the excluded OTU
    with zero_value. Returns the data array and a list 
    [(OTU_name_1, OTU_name_2), ...] where each tuple indicates
    a set of OTUs that are mutually exclusive within each subject.

    """
    first_level_up = {n.up for n in otu_tree.get_leaves()}
    leaf_pairs = [tuple(l.name for l in n.children) for n in 
                  first_level_up]
    data = linear_test_data(N_subjects, N_otus, N_intervals, N_timepoints,
                            names = otu_tree.get_leaf_names())
    for i in xrange(N_subjects):
        excluded_otus = map(random.choice,leaf_pairs)
        for otu_name in excluded_otus:
            otu_index = data.variable_names.index(otu_name)
            data.X[i][otu_index,:] = zero_value
    return data, leaf_pairs
    
def unroll(dataset):
    """
    Extract one big data matrix from a Dataset object, assuming
    time points are the same for all subjects and no data points 
    are missing. 

    Returns: 
    X - n_subjects x (n_otus * n_timepoints) observation matrix
    y - (n_subjects,) array of booleans giving the outcome for each.
    t - (n_timepoints,) array of the observation times
    
    """
    n_subjects = dataset.n_subjects
    n_otus = dataset.n_variables
    timepoints = dataset.T[0]# again, assumed the same for all subjects
    n_timepoints = len(timepoints)
    X = np.zeros((n_subjects, n_otus*n_timepoints))
    for subject_index, subject_data in enumerate(dataset.X):
        X[subject_index,:] = subject_data.flatten()
    return X, dataset.y, timepoints

def filter_columns_by_variance(X, fraction):
    """ 
    Throw away covariates with low variance.

    Arguments:
    X - n_subjects x n_observations array
    fraction - number between 0 and 1
    Returns:
    Xprime: n_subjects x (fraction*n_observations) array containing
    columns from X with the largest variance
    keep_indices: vector of booleans indicating which indices are 
    kept (Xprime = X[:,keep_indices]). Allows the same filtering
    to be performed on unseen data with the same format as X. 

    In the unlikely case of exact ties in variance, the return 
    matrix may have more than fraction*n_observation columns.

    Note Xprime is technically a new view of the same data in 
    memory.

    """
    _ , n_obs = X.shape
    n_keep = np.floor(fraction*n_obs)
    variances = np.var(X,axis=0)
    sorted_variances = np.sort(variances)
    threshold_variance = sorted_variances[-1*n_keep]
    keep_indices = variances>=threshold_variance 
    Xprime = X[:,keep_indices]
    return Xprime, keep_indices 
    
