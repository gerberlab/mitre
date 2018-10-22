"""Subsampling of subjects and timepoints from an original dataset.

Usage
-----
This program can be run from the command line. The first command line argument
must contain the pickle file of the original dataset which will be subsampled.
To choose how many subjects and timepoints to subsample, use the arguments
--timepoints and --subjects followed by the desired numbers of timepoints and
subjects. The final optional argument is --filename_add, which can be used to
supply a string that will be appended to each filename when the subsampled
datasets are saved. To subsample only subjects or only timepoints, do not use
the other argument.

For example, to subsample master.pkl, selecting 48, 32, and 20 subjects, and
24, 18, 12, and 6 timepoints, run:

python subsample.py master.pkl --subjects 48 32 20 --timepoints 24 18 12 6
"""
import math
import itertools
import pickle
import argparse
import numpy as np
import scipy.spatial
import pandas as pd
from mitre.rules import Dataset


def subsample_dataset(master_dataset, subjects, timepoints):
    """Subsample particular subjects and timepoints from a MITRE dataset.

    Parameters
    ----------
    master_dataset : mitre.rules.Dataset
        Dataset containing the original subjects and timepoints.
    subjects : list of int
        Indices of the subjects to be kept in the subsampled dataset.
    timepoints : list of int
        Indices of the timepoints to be kept in the subsampled dataset.

    Returns
    -------
    mitre.rules.Dataset
        The subsampled dataset.
    """
    new_dataset = master_dataset.copy()

    if new_dataset.additional_covariate_matrix is not None:
        new_dataset.additional_covariate_matrix = \
                            master_dataset.additional_covariate_matrix[subjects]

    new_X = []
    new_T = []
    new_y = []
    new_subject_IDs = []
    for k in list(subjects):
        new_X.append(new_dataset.X[k])
        new_T.append(new_dataset.T[k])
        new_y.append(new_dataset.y[k])
        new_subject_IDs.append(new_dataset.subject_IDs[k])

    for k in range(len(new_y)):
        new_X[k] = new_X[k][:, timepoints]
        new_T[k] = new_T[k][list(timepoints)]

    new_dataset.X = new_X
    new_dataset.T = new_T
    new_dataset.y = np.array(new_y)
    new_dataset.subject_IDs = new_subject_IDs
    new_dataset.n_subjects = len(new_subject_IDs)
    if isinstance(new_dataset.subject_data, pd.DataFrame):
        new_dataset.subject_data = new_dataset.subject_data.loc[new_subject_IDs]
    new_dataset._primitive_result_cache = {}

    return new_dataset


def all_subsamples(master_dataset, new_subject_values, new_timepoint_values,
                   filename_suffix=''):
    """Subsample all sets of subjects and timepoints from a MITRE dataset.

    Starting with a master dataset with some number of subjects and timepoints,
    this function subsamples subjects and timepoints to generate new datasets
    in a grid of new values of subjects and timepoints. For multiple new values
    of subjects or timepoints, the subsampling is performed such that each
    smaller set of subjects or timepoints is a subset of the next largest set.

    The subsampled datasets are written as pickle files to the working
    directory. For each subsampled dataset, a text file indicating which
    subjects and timepoints have been selected for that dataset is also written.

    Parameters
    ----------
    master_dataset : mitre.rules.Dataset
        Dataset containing the original subjects and timepoints.
    new_subject_values : list of int
        The new numbers of subjects for the subsampled datasets.
    new_timepoint_values : list of int
        The new numbers of timepoints for the subsampled datasets.
    filename_suffix : str
        Optional string to add to the end of the filename for each pickle.
    """

    original_control_indices = np.where(master_dataset.y == False)[0]
    original_case_indices = np.where(master_dataset.y == True)[0]
    original_timepoint_indices = np.arange(len(master_dataset.T[0]))

    original_num_subjects = len(master_dataset.y)
    original_num_timepoints = len(master_dataset.T[0])

    new_subject_values.sort(reverse=True)
    new_timepoint_values.sort(reverse=True)

    # For each new number of subjects, randomly choose from the next largest
    # dataset half that number (rounded up) of controls and half (rounded down)
    # of cases.
    control_indices = [original_control_indices]
    case_indices = [original_case_indices]
    if len(new_subject_values) > 0:
        if max(new_subject_values) > original_num_subjects:
            raise ValueError('More subjects requested than present in master '
                             'dataset.')
        for i in range(len(new_subject_values)):
            new_controls = sorted(np.random.choice(control_indices[-1],
                                 int(math.ceil(new_subject_values[i]/2)), replace=False))
            new_cases = sorted(np.random.choice(case_indices[-1],
                               int(math.floor(new_subject_values[i]/2)), replace=False))
            control_indices.append(new_controls)
            case_indices.append(new_cases)

    timepoint_indices = [original_timepoint_indices]
    prev_timepoints = timepoint_indices[-1]
    if len(new_timepoint_values) > 0:
        if max(new_timepoint_values) > original_num_timepoints:
            raise ValueError('More timepoints requested than present in master '
                             'dataset.')

        for new_num_timepoints in new_timepoint_values:
            spacings = []
            subsets = []
            # For each possible subsample of the correct size, calculate all of the
            # pairwise distances within the subsample, and pick the one which is the
            # most evenly spaced.
            for subset in itertools.combinations(prev_timepoints,
                                                 new_num_timepoints):
                subsets.append(subset)
                subset = list(subset)
                subset = [-1] + subset + [original_num_timepoints+1]
                subset = np.array(subset)
                spacings.append((1/np.power(scipy.spatial.distance.pdist(
                                                  subset[:, np.newaxis]), 2)).sum())
            timepoint_indices.append(np.array(subsets[spacings.index(min(spacings))]))
            prev_timepoints = timepoint_indices[-1]

    for i, (controls, cases) in enumerate(zip(control_indices, case_indices)):
        for j, timepoints in enumerate(timepoint_indices):
            if i!=0 or j!=0:
                new_dataset = subsample_dataset(master_dataset,
                                                list(controls)+list(cases),
                                                timepoints)

                filename = 'subsampled_{}subj_{}pts{}_simulated_data_object.pickle'\
                                            .format(len(list(controls)+list(cases)),
                                                    len(timepoints),
                                                    filename_suffix)
                with open(filename, 'w') as f:
                    pickle.dump(new_dataset, f)

                filename = 'subsampled_{}subj_{}pts{}_simulated_data_info.txt'\
                                            .format(len(list(controls)+list(cases)),
                                                    len(timepoints),
                                                    filename_suffix)
                with open(filename, 'w') as f:
                    f.write(str(list(controls)) + '\n')
                    f.write(str(list(cases)) + '\n')
                    f.write(str(list(timepoints)) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Subsample MITRE dataset.')
    parser.add_argument('master_dataset', type=str, nargs='?')
    parser.add_argument('--subjects', nargs='*', type=int)
    parser.add_argument('--timepoints', nargs='*', type=int)
    parser.add_argument('--filename_add', nargs='?', type=str)

    args = parser.parse_args()

    subjects = args.subjects
    timepoints = args.timepoints
    filename_add = args.filename_add

    if subjects is None:
        subjects = []
    if timepoints is None:
        timepoints = []
    if filename_add is None:
        filename_add = ''

    print(subjects)
    print(timepoints)
    print(filename_add)

    with open(args.master_dataset, 'rb') as f:
        master_dataset = pickle.load(f)

    all_subsamples(master_dataset, subjects, timepoints, filename_add)
