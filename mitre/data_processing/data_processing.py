""" Utilities for preprocessing data and generating synthetic test data.

"""
import numpy as np
from rules import Dataset

def linear_test_data(subjects, variables, intervals, timepoints, names=None):
    if names is None:
        variable_names = ['var%d' % i for i in xrange(variables)]
    else:
        variable_names = names
    variable_prior = np.ones(variables)
    # Avoid timepoints exactly at t = 1.0 to avoid having to special case 
    # the definition of the relevant time intervals, usually [t0,t1)
    t = np.linspace(0,0.99,timepoints) 
    T = []
    X = []
    y = np.zeros(subjects) # Fill in the true y later depending on the model of interest
    for i in xrange(subjects):
        T.append(t.copy())
        subject_data = np.zeros((variables, timepoints))
        for j in xrange(intervals):
            relevant_indices = (float(j)/intervals <= t) & (t < (j+1.0)/intervals)
            center = (j+0.5)/intervals
            internal_t = t[relevant_indices] - center
            for k in xrange(variables):
                slope = np.random.uniform(low=-1.,high=1.)
                average = np.random.uniform(low=-1.,high=1.)
                subject_data[k, relevant_indices] = average + slope * internal_t
        X.append(subject_data)
    return Dataset(X,T,y, variable_names, variable_prior, 0, 1)

def exponentiate_dataset(data):
    new_data = data.copy()
    new_data.X = np.exp(data.X)
    return new_data

def log_transform_dataset(data,zero_data_offset=1e-6,zero_tolerance=1e-10):
    new_data = data.copy()
    # We expect the data to be positive, so don't take the absolute
    # value before checking to see what is close to zero
    new_data.X[new_data.X<zero_tolerance] = zero_data_offset
    new_data.X = np.log(new_data.X)
    return new_data


