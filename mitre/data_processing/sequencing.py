""" Methods to simulate 16S sequencing experiments.

"""
import numpy as np
from ..rules import Dataset
from .transforms import take_relative_abundance

def dirichlet_multinomial_sequencing(data, N_counts, dispersion):
    """ Simulate 16S amplicon sequencing via Dirichlet-multinomial dist'n.

    Arguments:
    data - Dataset object of raw OTU abundances (will be converted 
    to relative abundance.) 
    N_counts - simulated sequencing depth (number of reads per sample)
    dispersion - dispersion parameter for the Dirichlet multinomial
    distribution used to distribute counts among the OTUs. The expected 
    frequency parameters are the relative abundances for each sample. 
   
    The definition of the dispersion parameter here may be nonstandard--
    it's actually a precision parameter or concentration parameter...
    (TODO: specify the parameterization clearly/rename this.)
 
    Output:
    new_data - Dataset object, with new_data.X a list of arrays of 
    simulated sequencing counts assigned to each OTU. 
    
    """
    # Take the relative abundances; this is harmless if the
    # data are already given as relative abundances.
    new_data = take_relative_abundance(data)
    for subject in xrange(data.n_subjects):
        n_timepoints = len(new_data.T[subject])
        abundances = new_data.X[subject]
        count_array = np.zeros(abundances.shape)
        for i in xrange(n_timepoints):
            relative_frequencies = abundances[:,i]
            dirichlet_parameter = dispersion * relative_frequencies
            effective_probabilities = np.random.dirichlet(
                dirichlet_parameter
            )
            counts_sample_i = np.random.multinomial(
                N_counts,
                effective_probabilities
            )
            count_array[:,i] = counts_sample_i
        new_data.X[subject] = count_array
    return new_data
            
