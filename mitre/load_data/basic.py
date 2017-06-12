"""
Methods to load simple abundance tables, data about samples, etc. 

"""
import numpy as np
import pandas as pd
from ..rules import Dataset
from ..data_processing import filtering

dequote = lambda s: s.strip('"')

def load_abundance_data(filename, strip_quotes=True):
    """ Read a CSV of abundances, returning a data frame.

    We expect a CSV formatted like:
    
    "", "GCA...GGG", "GCA...TGG"
    "sampleID1", 5, 0
    "sampleID2", 400, 25
    
    (though the data need not be integers.) 

    A pandas dataframe is returned. 
    By default, (double) quotes are stripped from the sample names
    and OTU identifiers.

    """

    # It is important that the sample IDs, thus the 
    # index column of this dataframe, be strings, 
    # so we specify a converter function for column 0.
    df = pd.read_csv(filename,index_col=0,
                     converters={0:lambda x: str(x)})
    if strip_quotes:
        df.rename(dequote, dequote)
    return df

def fasta_to_dict(filename):
    """ Read a mapping of IDs to sequences from a fasta file.

    For relabeling DADA2 RSVs. Returns a dict {sequence1: name1, ...}.

    """
    with open(filename) as f:
        whole = f.read()
    pairs = whole.split('\n>')
    table = {}
    # remove leading '>'
    pairs[0] = pairs[0][1:]
    for pair in pairs:
        name, sequence = pair.strip().split('\n')
        table[sequence] = name
    return table

def load_sample_metadata(filename, strip_quotes=True):
    """ Read subject, time, and optionally other data.

    Expects a three-column CSV with no header row.

    First column the sample ID, second column a unique identifier
    (treated as a string) for each subject, third column an observation
    time (treated as a float.)

    In the future, additional time-varying data measured at each
    sampling point will be accepted as additional columns.

    If strip_quotes is True (the default), remove quotation marks
    around the sample IDs if there are any.

    Returns a list of tuples, one for each subject:
    [(subject_identifier, [(timepoint1, sampleID1), ... ]), ...]
    
    To match pandas behavior, if _all_ subject identifiers appear
    to be integers, they are cast to int (from str.) 

    """
    # In the future load a data frame and iterate through the rows.
    table = {}
    with open(filename) as f:
        for line in f:
            sample, subject, time = line.strip().split(',')
            time = float(time)
            if strip_quotes:
                sample = dequote(sample)
            table.setdefault(subject,[]).append((time, sample))
    try:
        table = {int(k): v for k,v in table.iteritems()}
    except ValueError:
        pass
    for subject in table:
        table[subject].sort()
    return sorted(table.items())

def load_subject_data(filename):
    """ Read non-time-dependent data for each subject from csv.

    The first column must be the subject identifier used
    in the sample metadata file. The first row should give 
    field names (field name for the first column is not important.) 

    """
    df = pd.read_csv(filename,index_col=0)
    return df

def load_dada2_result(abundance_data_filename, 
                      sample_metadata_filename, subject_data_filename,
                      sequence_id_filename=None,
                      **kwargs):
    abundances = load_abundance_data(abundance_data_filename)
    if sequence_id_filename is not None:
        sequence_names = fasta_to_dict(sequence_id_filename)
        abundances = abundances.rename({}, sequence_names)
    sample_metadata = load_sample_metadata(sample_metadata_filename)
    subject_data = load_subject_data(subject_data_filename)
    return combine_data(abundances, sample_metadata,
                        subject_data,
                        **kwargs)

def combine_data(abundance_data,
                 sample_metadata,
                 subject_data,
                 experiment_start=None,
                 experiment_end=None,
                 outcome_variable=None,
                 outcome_positive_value=None,
                 additional_subject_categorical_covariates=[],
                 additional_covariate_default_states=[],
                 additional_subject_continuous_covariates=[]):
    """ Assemble data loaded from files into appropriate Dataset.

    abundance_data (a pandas.Dataframe), sample_metadata (a list),
    and subject_data (a pandas.Dataframe) should be formatted
    as the output of load_abundance_data, load_sample_metadata,
    and load_subject_data.

    If experiment_start is given, it will be passed to the Dataset
    constructor; otherwise the minimum observation time is used.
    Similarly for experiment_end.

    If outcome_variable and outcome_positive_value are given,
    subjects where subject_data[outcome_variable] == outcome_positive_value
    have y=1, and subjects where subject_data[outcome_variable] is NaN
    are excluded. Otherwise, y=0 for all subjects.

    If additional_subject_categorical_covariates are given, subjects
    where subject_data[variable] is NaN for any variable in the list
    are excluded. The same is done for additional_subject_continuous_covariates.

    The variable_prior is set to 1.0 for each variable.

    """

    # Assemble X and T and the list of subject IDs.
    X = []
    T = []
    subject_IDs = []
    skipped_samples = []
    for subject, samples in sample_metadata:
        subject_IDs.append(subject)
        this_subject_timepoints = []
        this_subject_observations = []
        for timepoint, sample_id in samples:
            try:
                values = abundance_data.loc[sample_id].values
            except KeyError:
                skipped_samples.append(sample_id)
                continue
            this_subject_timepoints.append(timepoint)
            this_subject_observations.append(
                abundance_data.loc[sample_id].values
            )
        T.append(np.array(this_subject_timepoints))
        # For whatever reason, convention is that
        # X entries have OTUs along first axis and
        # timepoints along second axis
        if this_subject_observations:
            X.append(np.vstack(this_subject_observations).T)
        else:
            X.append(np.array([]))
        
    # Extract variable names, count them, set up prior.
    variable_names = [s for s in abundance_data.columns]
    variable_prior = np.ones(len(variable_names))
    
    # Establish experiment start and end times.
    max_observation_times = []
    min_observation_times = []
    for timepoints in T:
        # Pathological subjects may have no observations
        if len(timepoints) == 0:
            continue
        max_observation_times.append(max(timepoints))
        min_observation_times.append(min(timepoints))
    if experiment_start is None:
        experiment_start = min(min_observation_times)
    if experiment_end is None:
        experiment_end = max(max_observation_times)

    if outcome_variable is not None and outcome_positive_value is not None:
        outcome_column = subject_data.loc[:,outcome_variable]
        sorted_outcomes = []
        for subject in subject_IDs:
            sorted_outcomes.append(
                outcome_column.loc[subject] == outcome_positive_value
            )
        y = np.array(sorted_outcomes)
        if len(y) != len(subject_IDs):
            raise ValueError('Wrong number of outcome values provided. Please check that the subject metadata file contains one line per subject.')
            
    else:
        y = np.zeros(len(subject_IDs))    
        
    result = Dataset(X,T,y,variable_names,variable_prior,
                     experiment_start, experiment_end,
                     subject_IDs, subject_data,
                     additional_subject_categorical_covariates = additional_subject_categorical_covariates,
                     additional_subject_continuous_covariates = additional_subject_continuous_covariates,
                     additional_covariate_default_states = additional_covariate_default_states
                     )

    if outcome_variable is not None:
        result = filtering.discard_where_data_missing(result,
                                                   outcome_variable) 
    for variable in additional_subject_categorical_covariates:
        result = filtering.discard_where_data_missing(result,
                                                      variable) 
    for variable in additional_subject_continuous_covariates:
        result = filtering.discard_where_data_missing(result,
                                                      variable) 
    # Note that the NaNs may have led to spurious or erroneous columns
    # in the covariate matrix which will go away if we regenerate it.
    if (additional_subject_categorical_covariates or
        additional_subject_continuous_covariates):
        result.generate_additional_covariate_matrix()
        
    return result
