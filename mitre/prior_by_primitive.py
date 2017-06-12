""" Estimate prior distribution over primitives.

This can eventually be made a method of the model class.

"""

import io
import logging
import scipy.stats
from scipy.stats import poisson, nbinom
import numpy as np
import logit_rules
from rules import logger

from tqdm import trange

# The estimation steps here are some of the most time consuming
# in our demo calculations, so we print a nice progress bar to confirm that
# something is happening. This needs to be done through the logging system,
# so that it will not show up when the global MITRE verbose option is not set.

class TqdmIO(io.StringIO):
    def __init__(self, logger):
        super(TqdmIO,self).__init__()
        self.logger = logger
        self.buf = ''
    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(logging.INFO, self.buf)


def estimate_primitive_priors_lognormal(model, N_samples):
    """ Estimate prior probabilities of choosing particular primitives to fill a slot.
    
    Marginalizes over the hyperparameters governing the primitive distribution
    using a Monte Carlo approach.
    
    Returns a dict mapping primitives (as tuples) to normalized  estimated prior probabilities.


    """
    N_primitives = len(model.rule_population)
    probabilities = np.zeros(N_primitives)
    # Set up a blank state; relevant attributes updated below
    state = logit_rules.LogisticRuleState( 
        rl=None, omega=None, beta=None, 
        phylogeny_mean=None, phylogeny_std=None,
        window_typical_fraction=None, window_concentration=None)
    N_errors = 0
    tqdm_file = TqdmIO(logger)
    for j in trange(N_samples, file = tqdm_file, mininterval=5):
        # Update the hyperparameters.
        # 
        # Phylogeny log mean, from a normal
        state.phylogeny_mean = (
            scipy.stats.norm.rvs(
                loc=model.phylogeny_lambda_l,
                scale=np.sqrt(model.phylogeny_mean_hyperprior_variance)
            )
        )
        # Phylogeny std, from uniform
        state.phylogeny_std = (
            np.random.rand() * model.phylogeny_std_upper_bound
        )
        # Window typical fraction, from a uniform distribution on the
        # unit interval:
        state.window_typical_fraction = np.random.rand()
        # Window concentration, from an exponential
        state.window_concentration = (
            scipy.stats.expon.rvs(
                scale = model.window_concentration_typical
            )
        )
        # Sometimes we can get a division by zero that we will not 
        # (as likely) see during sampling...
        try:
            probability_sample_j = model.flat_prior(state)
        except ZeroDivisionError:
            N_errors += 1
            continue
        # It is on a log scale.
        probabilities += np.exp(probability_sample_j)
    if N_errors > 0.01 * N_samples:
        raise Exception('Too many zero division errors, exiting')
    return dict(zip(model.rule_population.flat_rules, probabilities / float(N_samples)))

####
# Similarly we can estimate the distribution of lengths. This doesn't 
# conceptually belong in a file by this name but this will all be made part of the model class 
# in a later release.
def estimate_length_distribution(model, N_samples):
    """ Estimate prior probability of rule list containing a particular number of primitives in total.
    
    Marginalizes over the hyperparameters governing the length distribution.
    using a Monte Carlo approach.
    
    Returns a vector with v[0] = estimated probability of length 0, etc., up to 
    v[max_length]. 

    Note the truncation of the negative binomial distributions is handled in a rather
    naive way, such that if the maximum lengths are comparable to the typical values from
    the non-truncated distrubtion, or are smaller, this function will be very, very slow.

    """
    # First do the probabilities conditional on nonzero overall length; adjust below
    max_length = model.max_primitives * model.max_rules
    nonzero_probabilities = np.zeros(max_length + 1)
    excess = 0
    # Truncation makes it tedious for us to try to work with the length probabilities 
    # analytically, so we do this by simulation. It's cheap.
    nb = lambda alpha, beta: nbinom(alpha, beta/(beta+1.0))
    excess_m_distribution = nb(model.hyperparameter_alpha_m, model.hyperparameter_beta_m)
    excess_primitive_distribution = nb(model.hyperparameter_alpha_primitives, model.hyperparameter_beta_primitives)

    tqdm_file = TqdmIO(logger)
    for m in trange(1, model.max_rules, file=tqdm_file, mininterval=5):
        probability_m = (excess_m_distribution.pmf(m-1) /
                         excess_m_distribution.cdf(model.max_rules - 1))
        counts_given_m = np.zeros(max_length + 1)
        for j in xrange(N_samples):
            excess_lengths_each_rule = np.ones(m) * model.max_primitives
            for k in xrange(m):
                while excess_lengths_each_rule[k] >= model.max_primitives:
                    excess_lengths_each_rule[k] = excess_primitive_distribution.rvs()
            total_primitives = np.sum(excess_lengths_each_rule) + m
            counts_given_m[int(total_primitives)] += 1
        distribution_given_m = counts_given_m / float(N_samples)
        nonzero_probabilities += probability_m * distribution_given_m

    zero_probability = (model.hyperparameter_a_empty / 
                        (model.hyperparameter_a_empty + model.hyperparameter_b_empty))
    probabilities = np.zeros(max_length+1)
    probabilities[0] = zero_probability
    probabilities[1:] = (1.0-zero_probability) * nonzero_probabilities[1:]
    return probabilities
        
