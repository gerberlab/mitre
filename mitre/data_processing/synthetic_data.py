""" Methods for generating synthetic test data.

Most return a rules.Dataset object. Typically these methods only
simulate the microbiome data; synthetic outcome generation is left to
other code. 

"""
import numpy as np
from . import transforms, sequencing
from ..rules import Dataset


def linear_test_data(subjects, variables, intervals, timepoints, names=None):
    if names is None:
        variable_names = ['var%d' % i for i in xrange(variables)]
    else:
        variable_names = names
        assert len(variable_names) == variables
    # By default, set a uniform prior.
    variable_prior = np.ones(variables)/variables 
    # Avoid timepoints exactly at t = 1.0 to avoid having to special case 
    # the definition of the relevant time intervals, usually [t0,t1)
    t = np.linspace(0,0.99,timepoints) 
    T = []
    X = []
    # Fill in the true y later depending on the model of interest
    y = np.zeros(subjects) 
    for i in xrange(subjects):
        T.append(t.copy())
        subject_data = np.zeros((variables, timepoints))
        for j in xrange(intervals):
            relevant_indices = ((float(j)/intervals <= t) &
                                (t < (j+1.0)/intervals))
            center = (j+0.5)/intervals
            internal_t = t[relevant_indices] - center
            for k in xrange(variables):
                slope = np.random.uniform(low=-1.,high=1.)
                average = np.random.uniform(low=-1.,high=1.)
                subject_data[k, relevant_indices] = (
                    average + slope * internal_t
                )
        X.append(subject_data)
    return Dataset(X,T,y, variable_names, variable_prior, 0, 1)

# In the longer term we want to not know exactly where the 
# boundaries are... 
# Draw them from a Dirichlet distribution?

def basic_counts_simulator(subjects, 
                           variables, 
                           intervals, 
                           timepoints, 
                           names=None,
                           depth=10000.,
                           dm_parameter=281.,
                           tree=None):
    """ Simulate random counts data.

    First, true abundances are simulated by calling linear_test_data
    with the appropriate subset of the arguments, then exponentiating.
    If tree is not None, sister leaves are made mutually exclusive.

    Then, counts are simulated, starting from the relative abundances,
    to the specified depth, with the specified Dirichlet-multinomial
    concentration parameter.

    Returns two Datasets: counts and true_abundances. 

    """
    base_trends = linear_test_data(subjects, 
                                   variables,
                                   intervals,
                                   timepoints,
                                   names=names)
    true_abundances = transforms.exponentiate(base_trends)
    if tree is not None:
        true_abundances = transforms.make_sister_leaves_mutually_exclusive(
            true_abundances, 
            tree
        )

    counts = sequencing.dirichlet_multinomial_sequencing(
        true_abundances,
        N_counts = depth,
        dispersion = dm_parameter
    )
    return counts, true_abundances
