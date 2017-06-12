"""
Rule-based logistic regression models with Polya-Gamma sampling.

"""
import numpy as np
import rules as rules
from rules import PrimitiveRule, RuleList, DiscreteRulePopulation, Dataset, DiscreteRuleModel, choose_from_relative_probabilities, RuleListSampler, logger, match_pairs, match_features, primitive_similarity_score
from scipy.stats import multivariate_normal, beta, poisson, nbinom
import scipy.stats
from scipy.special import gammaln
from scipy.misc import logsumexp
from mvnpdf_cholesky import mc_logpdf_np
import pypolyagamma
import multiprocessing as mp
from Queue import Empty
import time
import random
import efficient_likelihoods

default_n_workers = 6

def _likelihood_worker(model, omega, truth_table, X, i, 
                       argument_queue, result_queue):
    
    """ Worker function for parallelizing the evaluation of many likelihoods.

    Evaluates the likelihood of the effective response z (a function of omega)
    given various values of the covariate matrix X, formed by substituting in
    columns from truth_table at position i.

    For use with multiprocessing.Process.

    Arguments:
    model - LogisticRuleModel whose likelihood to evaluate
    omega - vector of auxiliary variables
    truth_table - list-like collection of vectors of truth values of rules
    X - design matrix
    i - column of design matrix to substitute 
    argument_queue: each item should be a tuple (start, stop) or the
    string 'end'; this worker will then substitute
    truth_table[start:stop] and evaluate and return those likelihoods,
    unless the item is 'end', in which case it will return.
    result_queue: each entry will be a list [start_index, array_of_likelihoods]

    """
    done = 0
    while True:
        item = argument_queue.get()
        if item == 'end':
#            print 'Worker finishing after completing %d tasks' % done
            return
        start, stop = item

        subtable = truth_table[start:stop,:] # view, not copy
        n, _ = subtable.shape
        likelihoods = np.zeros(n)
        for output_index, truth_vector in enumerate(subtable):
            X[:,i] = truth_vector
            likelihoods[output_index] = model.likelihood_z_given_X(omega, X)
        result_queue.put([start, likelihoods])
        done += 1
            
ppg = pypolyagamma.PyPolyaGamma(0) # TODO better initialization

def log_multinomial_coefficient(list_of_primitive_rules):
    """ 
    Count copies of each primitive and calculate a log multinomial coefficient.
~    
    """
    list_of_tuples = [p.as_tuple() for p in list_of_primitive_rules]
    counts = dict.fromkeys(set(list_of_tuples),0)
    for t in list_of_tuples:
        counts[t] += 1
    N = len(list_of_primitive_rules)
    v = np.array(counts.values())
    return gammaln(N+1)-np.sum(gammaln(v+1))

class LogisticRuleModel(DiscreteRuleModel):
    """ Discretized rule list model with logistic regression likelihood.

    This class inherits the base Poisson priors on the rule list structure
    from the DiscreteRuleModel.

    """
    def __init__(self, data, tmin,
                 prior_coefficient_variance=100.,
                 hyperparameter_alpha_m=0.5, 
                 hyperparameter_beta_m=2.0, 
                 hyperparameter_alpha_primitives=2.0,
                 hyperparameter_beta_primitives=4.0,
                 window_concentration_typical=5.0,
                 window_concentration_update_ratio=0.2,
                 n_workers=default_n_workers,
                 N_intervals=10, 
                 hyperparameter_a_empty=0.5,
                 hyperparameter_b_empty=0.5,
                 min_phylogeny_delta_l=1e-3,
                 max_rules = 10,
                 max_primitives = 10,
                 delta_l_scale_mean = 50.,
                 delta_l_scale_sigma = 25.,
                 lambda_l_offset = 0.,
                 **kwargs):
        # Do not call the superclass constructors: we lack some
        # arguments (parameters) they expect.
        self.data = data
        self.tmin = tmin
        self.rule_population = DiscreteRulePopulation(
            self, self.tmin, N_intervals,
            max_thresholds=kwargs.get('max_thresholds'),
            tmax = kwargs.get('tmax',None)
        )
        self.data._primitive_result_cache = self.rule_population.truth_table
        
        self.prior_coefficient_variance = prior_coefficient_variance
        self.hyperparameter_alpha_m = hyperparameter_alpha_m 
        self.hyperparameter_beta_m = hyperparameter_beta_m
        self.hyperparameter_alpha_primitives = hyperparameter_alpha_primitives 
        self.hyperparameter_beta_primitives = hyperparameter_beta_primitives
        self.n_workers = n_workers

        self.max_rules = max_rules
        self.max_primitives = max_primitives

        self.hyperparameter_a_empty=hyperparameter_a_empty
        self.hyperparameter_b_empty=hyperparameter_b_empty

        # 2.5-100 precentile range: we may have outliers at the low
        # end, but the high end is the root of the tree, and fairly
        # stable
        phylogeny_low, phylogeny_high = np.percentile(
            np.log(self.data.variable_weights),
            [2.5, 100.0]
        )
        # When we don't want to take phylogeny information into
        # account, we typically assign all variables equal weights;
        # but then delta_l ~ 0, which leads to pathological behavior; 
        # so we establish some small finite minimum value.
        self.phylogeny_delta_l = max(phylogeny_high - phylogeny_low,
                                     min_phylogeny_delta_l)
        self.phylogeny_lambda_l = (
            np.median(np.log(self.data.variable_weights)) + 
            lambda_l_offset
        )
        self.phylogeny_mean_hyperprior_variance = (
            delta_l_scale_mean * self.phylogeny_delta_l ** 2
        ) 
        self.phylogeny_std_upper_bound = (
            delta_l_scale_sigma * self.phylogeny_delta_l
        ) 

        self.window_concentration_typical = window_concentration_typical
        self.window_concentration_update_ratio = window_concentration_update_ratio

        self.window_duration_epsilon = 0.01 # epsilon_w

    def likelihood_z_given_rl(self, omega, rl):
        """ Log-likelihood of working response kappa/omega given rl.

        Integrates out beta.

        I think this terminology is somewhat misleading and will revise it.

        """

        X = self.data.covariate_matrix(rl)
        return self.likelihood_z_given_X(omega,X)

    def likelihood_z_given_X(self, omega, X):
        k, _ = X.shape
        kappa = self.data.y - 0.5
        z = kappa/omega
        covariance_z = np.diag(1./omega) + (self.prior_coefficient_variance * 
                                            np.dot(X,X.T))
        return mc_logpdf_np(z,covariance_z)

    def likelihood_y_given_X_beta(self, X, beta):
        psi = np.dot(X,beta)
        probabilities = 1.0/(1.0+np.exp(-1.0*psi))
        return np.sum(
            np.log(np.where(self.data.y,probabilities,1.0-probabilities))
        )

    def likelihoods_of_all_options(self, rl, current_omega, i,j,
                                   N_workers=None):
        population = self.rule_population
        n_primitives = len(population)
        likelihoods = np.zeros(len(population))
        n_subjects = self.data.n_subjects

        # Make sure this is a sensible position to update. 
        if (i >= len(rl.rules)) or (j >= len(rl.rules[i])): 
            raise ValueError('No position (%d,%d) in rule list.' % (i,j))
        
        # Evaluate the other primitives in this rule. 
        this_rule_other_primitives = (
            [self.data.apply_primitive(p) for p in rl.rules[i][:j]] + 
            [self.data.apply_primitive(p) for p in rl.rules[i][j+1:]] 
        )
        # Use a default value in the reduce() call for cases where
        # there are no other primitives in the rule.
        this_rule_other_primitives = reduce(np.logical_and,
                                            this_rule_other_primitives,
                                            np.ones(n_subjects,dtype='bool')
                                           )

        # What are the truth values for this rule, substituting in
        # every primitive?
        # (N_pop x N_subjects) = (N_pop x N_subjects) .* (N_subjects x 0)
        # Need to be careful here, as the dot product of two arrays of
        # booleans is also boolean (i.e., in this context, wrong)!
        this_rule_all_options = (
            population.flat_truth * this_rule_other_primitives
        )
        this_rule_all_options = this_rule_all_options.astype(np.float64)

        base_rl = rl.copy()
        del base_rl[i]
        X = self.data.covariate_matrix(base_rl)
        kappa = self.data.y - 0.5
        response = kappa/current_omega

        likelihoods = efficient_likelihoods.likelihoods_of_all_options(
            X.astype(np.float64), current_omega, response, this_rule_all_options,
            np.float64(self.prior_coefficient_variance)
            )
        return likelihoods


    def prior(self, state):
        """ Evaluate overall prior probability of a state.

        Takes into account the rule list as well as phylogeny
        and list-length hyperparameters, etc. 
        
        Log scale.

        """
        terms = [
            self.prior_contribution_hyperpriors(state),
            self.rule_list_prior(state),
            self.prior_contribution_coefficients(state)
        ]
        return np.sum(terms)

    def prior_contribution_coefficients(self, state):
        """ Calculate prior contribution from regression coefficients.

        Log scale.

        """
        # This will need to be revised if we allow different 
        # variances for different classes of coefficients, eg 
        # microbiome related and host-covariate related
        dimensions = len(state.beta)
        normalization = -0.5*dimensions*(
            np.log(2.0*np.pi*self.prior_coefficient_variance) 
        )
        exponent = (-0.5*np.dot(state.beta, state.beta) / 
                     (self.prior_coefficient_variance))

        return normalization + exponent

    def prior_contribution_hyperpriors(self, state):
        """ Add up prior contributions from parameters governed by hyperpriors
        
        Log scale.

        """
        terms = [
            self.prior_contribution_phylogeny_parameters(state),
            self.prior_contribution_window_parameters(state),
            self.prior_contribution_empty_probability(state),
        ]
        return np.sum(terms)

    def prior_contribution_empty_probability(self, state):
        return scipy.stats.beta.logpdf(
            state.empty_probability, 
            self.hyperparameter_a_empty,
            self.hyperparameter_b_empty
        )
 
    def prior_contribution_window_parameters(self, state):
        """ Calculate prior contribution from window length parameters.

        Log scale.

        """
        # the prior on window_typical_fraction is uniform, so it
        # does not contribute.
        
        return scipy.stats.expon.logpdf(
            state.window_concentration,
            scale = self.window_concentration_typical
        )

    def prior_contribution_phylogeny_parameters(self, state):
        """ Evaluate prior probability of phylogeny mean/std

        Log scale.

        """
        mean_prior = scipy.stats.norm.logpdf(
            state.phylogeny_mean,
            loc = self.phylogeny_lambda_l,
            scale = np.sqrt(self.phylogeny_mean_hyperprior_variance)
        )
        if (0. <= state.phylogeny_std and
            state.phylogeny_std <= self.phylogeny_std_upper_bound):
            std_prior = -1.0*np.log(self.phylogeny_std_upper_bound)
        else:
            std_prior = -np.inf
        return mean_prior + std_prior


    def rule_list_prior(self, state):
        """ Evaluate prior log-probability of a rule list in context.
        
        Differs from the base class in that it takes into account the
        contribution from the multinomial coefficients.

        """
        # When last checked, only the move proposal called this
        # method.
        rule_list = state.rl
        primitive_piece = self.prior_contribution_from_primitives(state)
        structure_piece = self.structure_prior(
            rule_list,
            empty_probability=state.empty_probability,
        )
        multinomial_piece = self.multinomial_prior(rule_list)
        return primitive_piece + structure_piece + multinomial_piece

    def structure_prior(self, rule_list, empty_probability):
        """ Evaluate the log prior probability of a rule list structure.

        Assumes truncated negative binomial priors on the overall and subrule
        lengths (strictly those lengths minus 1) if the list is not
        empty, and an overall prior probability of the list being
        empty. If the length of the whole list, or any of the compound
        rules, is greater than the maximal values, the 
        log-probability is negative infinity.

        Properly normalized.

        """

        # We parameterize the negative binomial distribution
        # differently from scipy.stats (consistently, we work
        # with the parameterization in which the mean is alpha/beta)
        # Technically these are the _excess_ sublength 
        # distributions but putting that in the variable name
        # gets cumbersome.
        length_distribution = nbinom(
            self.hyperparameter_alpha_m, 
            self.hyperparameter_beta_m / (
                1.0 + self.hyperparameter_beta_m
            )
        )
        log_rule_normalization = length_distribution.logcdf(
                self.max_rules - 1
        )

        sublength_distribution = nbinom(
            self.hyperparameter_alpha_primitives, 
            self.hyperparameter_beta_primitives / (
                1.0 + self.hyperparameter_beta_primitives
            )
        )
        log_primitive_normalization = (
            sublength_distribution.logcdf(
                self.max_primitives - 1
            )
        ) 


        rules = rule_list.rules
        if len(rules) == 0:
            return np.log(empty_probability)
        else:
            # First check that the rule is valid
            if len(rules) > self.max_rules:
                return -np.inf
            sublengths = np.array([len(rule) for rule in rules])
            if np.max(sublengths) > self.max_primitives:
                return -np.inf
            
            l0 = np.log(1-empty_probability)      
            l1 = (
                length_distribution.logpmf(len(rules)-1) -
                log_rule_normalization
             )
            l2 = (
                np.sum(
                    sublength_distribution.logpmf(sublengths-1)
                ) -
                len(rules) * log_primitive_normalization
            )
        return l0 + l1 + l2


    def prior_contribution_from_primitives(self, state):
        """ Get contribution to log prior of this rl from choice of primitives.

        """
        flat_prior = self.flat_prior(state)
        rule_list = state.rl
        return sum(
            [flat_prior[self.rule_population.get_primitive_index(p)] for
             rule in rule_list for p in rule]
        )

    def multinomial_prior(self, rule_list):
        """ Log-prior contribution from multinomial coefficients. 

        """
        multinomial_piece = sum([log_multinomial_coefficient(subrule) for 
                                 subrule in rule_list.rules])
        return multinomial_piece

    def flat_prior(self, state):
        """ Evaluate log-probability of each primitive in the population.

        Return value is properly normalized.
        
        """
        raw_phylogeny_weights = self.rule_population.flat_variable_weights 
        phylogeny_weights = scipy.stats.norm.logpdf(
            np.log(raw_phylogeny_weights),
            loc = state.phylogeny_mean,
            scale = state.phylogeny_std
        )

        total_duration = float(self.data.experiment_end - self.data.experiment_start)
        durations = (self.rule_population.flat_durations /
                     ((1.0+self.window_duration_epsilon)*total_duration)
                    )
        window_a = (
            state.window_concentration *
            state.window_typical_fraction
        )
        window_b = (
            state.window_concentration *
            (1.0-state.window_typical_fraction)
        )
        window_weights = scipy.stats.beta.logpdf(
            durations, 
            window_a,
            window_b
        )

        weights = phylogeny_weights + window_weights
        normalization = logsumexp(weights)
        return weights - normalization

class LogisticRuleModelPriorOnly(LogisticRuleModel):
    def __getattribute__(self, name):
        if name.startswith('lambda'):
            raise ValueError('%s: %.3g' % (name, object.__getattribute__(self,name)))
        else:
            return object.__getattribute__(self, name)
        

    def likelihood_z_given_X(self, omega, X):
        return 0.

    def likelihood_y_given_X_beta(self, X, beta):
        return 0.

    def likelihoods_of_all_options(self, rl, current_omega, i,j,
                                   N_workers=default_n_workers):
        population = self.rule_population
        likelihoods = np.zeros(len(population))
        return likelihoods

class LogisticRuleState:
    """ Collect the things the sampler updates. """
    def __init__(self, rl, omega, beta, 
                 phylogeny_mean, phylogeny_std,
                 window_typical_fraction=0.5, window_concentration = 2.0,
                 empty_probability=0.5):
        self.rl = rl
        self.omega = omega
        self.beta = beta
        self.phylogeny_mean = phylogeny_mean
        self.phylogeny_std = phylogeny_std 
        self.window_typical_fraction = window_typical_fraction
        self.window_concentration = window_concentration
        self.empty_probability = empty_probability

    def copy(self):
        return LogisticRuleState(
            self.rl.copy(), 
            self.omega.copy(), self.beta.copy(),
            self.phylogeny_mean, self.phylogeny_std,
            self.window_typical_fraction, self.window_concentration,
            self.empty_probability
        )

class LogisticRuleSampler(RuleListSampler):
    def __init__(self, model, r0, local_updates_per_structure_update=10):
        self.model = model
        # Ensure r0 is sorted within each sublist
        r0 = r0.copy()
        for i in xrange(len(r0)):
            model.rule_population.sort_list_of_primitives(r0[i])
        # This may be a bad choice of omega_0...
        omega = np.ones(self.model.data.y.shape) 
        # Come up with some sensible initial values for the various
        # structure parameters
        phylogeny_mean = self.model.phylogeny_lambda_l
        phylogeny_std = min(self.model.phylogeny_std_upper_bound,
                            5.0*self.model.phylogeny_delta_l)
        self.current_state = LogisticRuleState(
            r0,omega,np.zeros(len(r0)+model.data.n_fixed_covariates),
            phylogeny_mean, phylogeny_std
         )
        # Could include X in the state except we don't really want to 
        # track it over the whole sampling process; definitely do 
        # need it to persist between subparts of iterations at least, though
        self.current_X = self.model.data.covariate_matrix(r0)
        # We specified an arbitary beta: we will overwrite it when we draw
        # it from its conditional in the first iteration. Because this
        # is not a valid value of beta, we don't add the initial state
        # to self.states.
        self.initial_state = self.current_state.copy()
        self.states = []
        ### The following are not updated after updating beta/z, but are 
        # after every change or attempted change to the rule list structure.
        self.likelihoods = []
        self.priors = [] 
        self.comments = []
        self.attempts = {}
        self.successes = {}
        self.local_updates_per_structure_update = (
            local_updates_per_structure_update
            )
        # We do want some better resolution on the distribution of 
        # coefficients for each state, so we'll store the interstitial beta
        # samples before each structure update step:
        self.additional_beta_values = []

        ### These don't conceptually need to be attributes of the sampler
        # object but this way we avoid recalculating them every iteration
        self.phylogeny_mean_proposal_std = (
            0.5*self.model.phylogeny_delta_l
        )
        self.phylogeny_std_proposal_std = (
            0.5*self.model.phylogeny_delta_l
        )
        self.window_fraction_proposal_std = (
            self.model.tmin /
            (self.model.data.experiment_end - 
             self.model.data.experiment_start)
        )
        self.window_concentration_proposal_std = (
            self.model.window_concentration_typical *
            self.model.window_concentration_update_ratio
        )

    def ensure_current_rl_sorted(self):
        rl = self.current_state.rl
        for i in xrange(len(rl)):
            self.model.rule_population.sort_list_of_primitives(rl[i])

    def step(self):
        beta_values = []
        for i in xrange(self.local_updates_per_structure_update):
            self.update_omega()
            self.update_beta()
            beta_values.append(self.current_state.beta)
        self.additional_beta_values.append(beta_values)
        move_type_indicator = np.random.rand()
        if move_type_indicator < 0.45:
            self.update_primitives()
        elif move_type_indicator < 0.9:
            self.add_remove()
        else:
            self.move_primitive()
        # If the structure moves are accepted, beta becomes out of date, 
        # perhaps grievously, so we defensively update it before
        # recording the new state
        self.update_beta()
        self.update_empty_probability()
        self.update_window_parameters()
        # As a convenience, have update_phylogeny_parameters return
        # the resulting state's prior value (which it calculates
        # anyway) and record that rather than recalculating it.
        prior = self.update_phylogeny_parameters()
        self.states.append(self.current_state.copy())
        self.likelihoods.append(
            self.model.likelihood_y_given_X_beta(self.current_X, 
                                                 self.current_state.beta)
        )
        self.priors.append(prior)

    def update_empty_probability(self):
        """ Update Theta_0 from its prior based on length of the rule list.

        """
        if len(self.current_state.rl) > 0:
            # effectively, the 'emptiness' trial has failed
            conditional_a = self.model.hyperparameter_a_empty
            conditional_b = self.model.hyperparameter_b_empty + 1
        else:
            conditional_a = self.model.hyperparameter_a_empty + 1
            conditional_b = self.model.hyperparameter_b_empty
        new_empty_probability = scipy.stats.beta.rvs(
            conditional_a,
            conditional_b
        )
        self.current_state.empty_probability = new_empty_probability
        
    def parameter_mh_update(self, proposed_state):
        """ Consider an MH step from self.current_state to proposed_state.

        This method is not a general-purpose Metropolis-Hastings updater;
        it simply consolidates the procedure used to update the 
        phylogeny and window prior parameters. In particular, it
        assumes that current_state and proposed_state differ only
        in their prior probability, not their likelihood.

        It is also assumed that the proposal distribution is symmetric.

        If the proposal is accepted, self.current_state is updated. 

        Returns the log prior associated with self.current_state
        after the update (whether this has changed or not.)

        """
        proposed_prior = self.model.prior(proposed_state)
        current_prior = self.model.prior(self.current_state)
        logger.debug('Parameter MH proposed/current priors: %.3g, %.3g' %
                     (proposed_prior, current_prior))
        # Both are on a log scale.
        # Note we assume this update cannot change the likelihood. 
        acceptance_probability = np.exp(proposed_prior - current_prior)
        logger.debug(
            'MH prior parameter update acceptance probability %.3f' %
            acceptance_probability
        )
        if np.random.rand() < acceptance_probability: 
            self.current_state = proposed_state
            logger.debug('Accepted.')
            return proposed_prior
        else:
            logger.debug('Rejected.')
            return current_prior

    def update_window_parameters(self):
        """ MH updates for time window fraction and concentration parameters.

        Lots of code overlap with update_phylogeny_parameters below:
        to be fixed.

        """
        # First, the typical window fraction.
        delta_f = self.window_fraction_proposal_std * np.random.randn()
        current_f = self.current_state.window_typical_fraction
        proposed_state = self.current_state.copy()
        # We want the increment of the fraction to 'bounce' off 
        # either end of the interval (0,1)-- more than once if necessary--
        # leading to this rather clumsy function:
        proposed_state.window_typical_fraction = np.abs(
            ((delta_f + current_f + 1.0) % 2.0) - 1.0
        )
        logger.debug(
            'Incrementing window f by %.3f from %.3f.' % 
            (delta_f, current_f)
        ) 
        self.parameter_mh_update(proposed_state)

        # Then the concentration parameter.
        delta_cw = self.window_concentration_proposal_std * np.random.randn()
        current_cw = self.current_state.window_concentration
        proposed_state = self.current_state.copy()
        proposed_state.window_concentration = np.abs(
            delta_cw + current_cw
        ) 
        logger.debug(
            'Incrementing window concenration by %.3f from %.3f.' %
            (delta_cw, current_cw)
        )
        self.parameter_mh_update(proposed_state)

    def update_phylogeny_parameters(self):
        """ MH updates for phylogenetic mean and std parameters. 

        For convenience, returns the prior probability after the MH 
        step is complete. 

        """
        # First, the mean.
        delta_mean = self.phylogeny_mean_proposal_std * np.random.randn()
        current_mean = self.current_state.phylogeny_mean
        proposed_state = self.current_state.copy()
        proposed_state.phylogeny_mean = (
            delta_mean + current_mean
        ) 
        logger.debug(
            'Incrementing phylogeny mean by %.3f from %.3f ' % 
            (delta_mean, current_mean)
        )
        self.parameter_mh_update(proposed_state)

        # Then (with unfortunate code duplication) the standard
        # deviation.
        delta_std = self.phylogeny_std_proposal_std * np.random.randn()
        current_std = self.current_state.phylogeny_std
        # It's possible to do this more efficiently, but the situation 
        # where this requires multiple iterations should arise rarely
        # with sensible parameter choices
        proposed_std = delta_std + current_std
        while (proposed_std < 0. or
               proposed_std > self.model.phylogeny_std_upper_bound):
            proposed_std = np.abs(proposed_std)
            if proposed_std > self.model.phylogeny_std_upper_bound:
                # Reflect off the upper bound
                proposed_std = (
                    2.0*self.model.phylogeny_std_upper_bound  -
                    proposed_std
                )
        proposed_state = self.current_state.copy()
        proposed_state.phylogeny_std = proposed_std
        logger.debug(
            'Incrementing phylogeny std by %.3f from %.3f to %.3f' % 
            (delta_std, current_std, proposed_std)
        )
        return self.parameter_mh_update(proposed_state)
    
    def sample(self, N):
        for i in xrange(N):
            logger.info('Step %d/%d' % (i,N))
            self.step()

    def sample_for(self, time_in_seconds):
        """ Keep sampling until a certain number of seconds has gone by.

        May overshoot somewhat (potentially a lot, if iterations are
        slow) as it will only stop the first time an iteration
        completes and more than time_in_seconds seconds have elapsed.

        """
        tstart = time.time()
        tstop = tstart + time_in_seconds
        i = 0
        while time.time() <= tstop:
            logger.info('Step %d' % i)
            self.step()
            i += 1

    def update_omega(self):
        """ Draw a new omega from its conditional distribution. """
        state = self.current_state 
        psi = np.dot(self.current_X, state.beta)
        ni = np.ones(psi.shape)
        new_omega = np.zeros(state.omega.shape)
        ppg.pgdrawv(ni,psi,new_omega)
        state.omega = new_omega

    def update_beta(self):
        """ Draw a new beta from its conditional distribution. """

        state = self.current_state
        # Note p may not equal len(self.current_state.beta)
        _, p = self.current_X.shape 
        v = self.model.prior_coefficient_variance
        kappa = self.model.data.y - 0.5
        
        # In later revisions we can avoid this inversion
        # CAUTION: the dot product here will be incorrect if current_X 
        # is stored as an array of Booleans rather than ints. 

        # the np.diag is wasteful here but okay for test purposes
        posterior_variance = np.linalg.inv(
            np.eye(p)/v + 
            np.dot(self.current_X.T, 
                   (np.dot(np.diag(state.omega),self.current_X))
                   )
        )
        posterior_mean = np.dot(posterior_variance,
                                np.dot(self.current_X.T, kappa)
                                )
        beta = np.random.multivariate_normal(posterior_mean,
                                             posterior_variance)
        state.beta = beta

########################################
########################################
########################################
##### CORE RL SAMPLING CODE

    def update_primitives(self, N_updates=None):
        """ Repeatedly choose position in list, sample over all options there.

        If N_updates is None, do a number of updates equal to the
        length of the list in primitives (though, because the target
        is random at each step, this does not necessarily amount to
        updating every primitive once.)

        """
        primitives = self.model.rule_population.flat_rules
        primitive_priors = self.model.flat_prior(self.current_state)
        rule_list = self.current_state.rl
        # Which combinations of rule i and primitive j are available?
        positions = []
        # How many copies of each present primitive are included in each rule?
        counts = []
        for i, rule in enumerate(rule_list):
            this_rule_counts = {}
            for j, primitive in enumerate(rule):
                positions.append((i,j))
                k = self.model.rule_population.get_primitive_index(primitive)
                this_rule_counts[k] = this_rule_counts.get(k,0) + 1
            counts.append(this_rule_counts)
        if N_updates is None:
            N_updates = 1 # len(positions)
        if len(positions) == 0:
            logger.info('No detectors to update')
            self.comments.append('Null update')
            return

        for _ in xrange(N_updates):
            i,j = random.choice(positions)
            primitive = rule_list[i][j]
            k0 = self.model.rule_population.get_primitive_index(primitive)
            # What are the relative multinomial pieces of the prior
            # for the rule lists we would obtain by putting primitives
            # 1...n_primitives at position i,j?
            multinomial_priors = np.zeros(len(primitives))
            # What are the relative rates of proposing an equivalent update step
            # for the rule lists we would obtain (etc).
            relative_rates = np.ones(len(primitives))
            # count the number of copies of each primitive in this
            # rule, except for the one at i,j
            base_counts = counts[i].copy()
            if k0 is not None:
                base_counts[k0] -= 1
            else:
                k0 = -1 # for later debug output
                logger.warn('Not decrementing base counts of atypical detector (should arise only in debugging)')
            
            for k, n in base_counts.iteritems():
                multinomial_priors[k] = -1.0*gammaln(n+1+1) # ie (n+1)!
                relative_rates[k] += n # note below this partially cancels the 
                # multinomial prior term and could be simplified.
            likelihoods = self.model.likelihoods_of_all_options(
                rule_list, self.current_state.omega, i, j
            )
            # print 'l'
            # print likelihoods
            # print 'primitive prior'
            # print primitive_priors
            # print 'm prior'
            # print multinomial_priors
            # Calculate posteriors on a log scale
            posteriors = likelihoods + primitive_priors + multinomial_priors 
            weights = posteriors + np.log(relative_rates)
            k = choose_from_relative_probabilities(np.exp(weights))
            logger.debug('Gibbs sweep for (%d,%d) chose option %d with '
                         'posterior %.3g (best was %.3g, typical %.3g)' % 
                         (i,j,k,posteriors[k],
                          max(posteriors),np.median(posteriors))
                        )
            rule_list[i][j] = PrimitiveRule(*primitives[k])
            logger.info('At (%d,%d) replacing %d with %d' % (i,j,k0,k))
            self.comments.append('At (%d,%d) replacing %d with %d' % (i,j,k0,k))
            counts[i][k] = counts[i].get(k,0)+1
            old_k0_counts = counts[i][k0]
            if old_k0_counts > 1:
                counts[i][k0] -= 1
            else:
                counts[i].pop(k0)
        # Sort _before_ recalculating X.
        self.ensure_current_rl_sorted()
        self.current_X = self.model.data.covariate_matrix(rule_list)
        # CAUTION: Beta is wrong.
    
    def add_remove(self):
        """ Add or remove primitive at randomly chosen position.

        """

        if np.random.rand() < 0.5:
            self.add()
        else:
            self.remove()
        self.update_beta()

    def consider_addition_to_new_rule(self, base_rule, position_to_add_at):
        """ Assess move from base_rule to a space of longer rule lists

        The space (S*) considered is the space of rule lists differing
        from the base rule lists by addition of a new length-1
        compound rule at position_to_add_at.

        Returns:
        new_rl - a new rule list with the appropriate structure,
           though the particular choice of primitive added is not 
           meaningful
        log_relative_prior - log(pi(S*)/pi(base_rule))
        log_relative_likelihood - log(L(S*|omega)/L(base_rule|omega))
        conditional_probabilities - array of length equal to 
           self.model.rule_population, giving (non-normalized) LOG
           probabilities of each candidate rule in S*, under the 
           posterior distribution conditional on omega restricted to S*.
        probability_remove_proposal - the probability that the corresponding
           removal will be proposed when generating a removal proposal
           from S*. (In this case this is simple, but 
           it is included for interface consistency with 
           consider_addition_to_existing_rule.)

        """
        mprime = len(base_rule) + 1
        new_rl = base_rule.copy()
        # Note that we insert an arbitrary primitive as a placeholder
        # This is important because likelihoods_of_all_options expects
        # a valid rule list as argument, and because we may want to be
        # able to do new_rl.copy(), etc. We also use this to calculate
        # the structure contribution to the prior of rule lists in S*,
        # which does not depend on the particular choice of addded
        # primitive
        placeholder = PrimitiveRule(*self.model.rule_population.flat_rules[0])
        new_rl.rules = (new_rl.rules[:position_to_add_at] + 
                        [[placeholder]] + 
                        new_rl.rules[position_to_add_at:])

        # log_relative_prior is log(C_1) in the notation of section 5.5
        structure_contributions = [
            self.model.structure_prior(
                rl,
                empty_probability=self.current_state.empty_probability,
                ) for
            rl in [base_rule, new_rl]
        ]
        log_relative_prior = (structure_contributions[1] - 
                              structure_contributions[0])
        
        old_X = self.model.data.covariate_matrix(base_rule)
        old_likelihood = self.model.likelihood_z_given_X(
            self.current_state.omega,
            old_X
        ) # log scale
        # Determine the likelihood of the new model, 
        # marginalizing over the choice of primitive at the insertion
        # position.
        priors = self.model.flat_prior(self.current_state)
        likelihoods = self.model.likelihoods_of_all_options(
            new_rl, 
            self.current_state.omega,
            position_to_add_at,
            0 # primitive index within rule
        )        
        primitive_distribution = priors + likelihoods
        log_marginalized_likelihood = logsumexp(primitive_distribution)
        log_likelihood_ratio = (log_marginalized_likelihood - 
                                old_likelihood)

        new_total_n_primitives = sum(map(len, new_rl))
        probability_remove_proposal = 1.0/new_total_n_primitives
        
        return (new_rl, 
                log_relative_prior,
                log_likelihood_ratio,
                primitive_distribution,
                probability_remove_proposal)
    
    def consider_addition_to_existing_rule(self, base_rule, 
                                           position_to_add_to):
        """ Assess move from base rule to space where one component is longer

        The space (S*) considered is the space of rule lists differing
        from the base rule list by addition of a new primitive rule to
        the composite rule at position_to_add_to.

        Returns:
        new_rl - a new rule list with the appropriate structure,
           though the particular choice of primitive added is not 
           meaningful
        relative_prior - pi(S*)/pi(base_rule) [not in log!]
        relative_likelihood - L(S*|omega)/L(base_rule|omega) [not in log!]
        conditional_probabilities - array of length equal to 
           self.model.rule_population, giving (non-normalized) LOG
           probabilities of each candidate rule in S*, under the 
           posterior distribution conditional on omega restricted to S*.
        probability_remove_proposal - the probability that the corresponding
           removal will be proposed when generating a removal proposal
           from S*. Here, we take the expected probability of proposing
           the corresponding removal, under the posterior distribution
           restricted to S*. This is T(S*) in the writeup.

        """
        base_subrule = base_rule[position_to_add_to]
        base_n_istar = len(base_subrule)
        base_rule_counts = {}
        for primitive in base_subrule:
            k = self.model.rule_population.get_primitive_index(primitive)
            base_rule_counts[k] = base_rule_counts.get(k,0) + 1
        # Every possible addition increases the _numerator_ of the
        # multinomial coefficient by a factor of (base_n_istar+1)
        log_relative_multinomial_coefficient = (
            np.log(base_n_istar + 1) + 
            np.zeros(len(self.model.rule_population.flat_rules))
        )
        # Changes to the denominator vary, however...
        for k, n in base_rule_counts.iteritems():
            log_relative_multinomial_coefficient[k] += (
                -1.0*np.log(n+1)
            )
        
        # relative_prior is C_2 in the notation of section 5.5.
        # We calculate the log here.
        new_rl = base_rule.copy()
        placeholder = PrimitiveRule(
            *self.model.rule_population.flat_rules[0]
        )
        new_rl[position_to_add_to].append(placeholder)
        
        structure_contributions = [
            self.model.structure_prior(
                rl,
                empty_probability=self.current_state.empty_probability,
                ) for
            rl in [base_rule, new_rl]
        ]
        log_relative_structure_prior = (structure_contributions[1] - 
                                        structure_contributions[0])

        primitive_prior = self.model.flat_prior(self.current_state)
        log_relative_prior_primitive_piece = (
            logsumexp(log_relative_multinomial_coefficient + 
                      primitive_prior) 
        )
        log_relative_prior = (
            log_relative_prior_primitive_piece + 
            log_relative_structure_prior
        )

        old_X = self.model.data.covariate_matrix(base_rule)
        old_likelihood = self.model.likelihood_z_given_X(
            self.current_state.omega,
            old_X
        ) # log scale
        # Determine the likelihood of the new model, 
        # marginalizing over the choice of primitive at the insertion
        # position.
        priors = primitive_prior
        likelihoods = self.model.likelihoods_of_all_options(
            new_rl, 
            self.current_state.omega,
            position_to_add_to,
            base_n_istar, # index where we have just added the primitive
        )        
        primitive_distribution = (
            priors + 
            log_relative_multinomial_coefficient + 
            likelihoods
        )
        log_marginalized_likelihood = (
            logsumexp(primitive_distribution) - 
            log_relative_prior_primitive_piece
        )

        log_likelihood_ratio = (log_marginalized_likelihood - 
                                old_likelihood)

        new_total_n_primitives = sum(map(len, new_rl))
        T_sstar_prefactor = (base_n_istar+1.0)/(new_total_n_primitives)
        # take the log, ie drop the exponential, here and the equiv above?
        T_sstar_other_factor = np.exp(
            logsumexp(
                priors +
                likelihoods
            )
            -
            logsumexp(
                log_relative_multinomial_coefficient +
                priors +
                likelihoods
            )
        )
        probability_remove_proposal = (
            T_sstar_prefactor *
            T_sstar_other_factor
        )
            
        return (new_rl, 
                (log_relative_prior), 
                log_likelihood_ratio, 
                primitive_distribution,
                probability_remove_proposal)

    def add(self, force_acceptance=False):
        current_rl = self.current_state.rl
        # Where to add? Can choose any of the existing rules.
        # a new rule before any existing rule, or a new rule 
        # at the end of the list. 
        starting_n_rules = len(current_rl)
        option = np.random.randint(2*starting_n_rules + 1)
        comment = 'add detector to '
        if option < starting_n_rules:
            # Use an existing rule. 
            rule_index = option
            # notionally we add to the end (the rule will
            # be sorted later)-- note the subrule will be
            # one longer than it currently is, so the index
            # of its last element is equal to the current length
            # (because of indexing from zero.)
            primitive_index = len(current_rl[rule_index]) 
            comment += 'rule %d' % rule_index
            consider_function = self.consider_addition_to_existing_rule
            debugcase = 1
        else:
            debugcase = 2
            # Insert a new length-1 rule:
            rule_index = option - starting_n_rules
            primitive_index = 0
            comment += 'new rule at position %d' % rule_index
            consider_function = self.consider_addition_to_new_rule

        result = consider_function(
            current_rl,
            rule_index
        )
        (new_rl, 
         relative_prior, 
         relative_likelihood,
         candidate_distribution,
         probability_remove_proposal) = result
        
        logger.debug('Adding to rule %d' % rule_index)
        index = lambda r: self.model.rule_population.get_primitive_index(r)
        if debugcase == 2:
            logger.debug('Rule is [] (new)')
        else:
            logger.debug('Rule is %s' % str(map(index, current_rl[rule_index])))
        logger.debug('Candidate distribution (log scale):')
        for i,value in enumerate(candidate_distribution):
            logger.debug('%d\t%.3f' % (i,value))

        probability_this_option = 1./(2.*len(current_rl) + 1.)
        probability_reverse_option = probability_remove_proposal

        logger.debug('Add block: %s' % comment)
        logger.debug('Prior, log(marg. new/old): %.3g' % 
                     relative_prior)
        logger.debug('Move choice prob (rev/for): %.3g/%.3g' % 
                     (probability_reverse_option, probability_this_option))
        logger.info('Likelihood, log(marg. new/old): %.3g' % 
                    relative_likelihood)
        acceptance_ratio = (
            np.exp(relative_prior+relative_likelihood) * 
            (probability_reverse_option / probability_this_option)
        )
        logger.debug('Acceptance ratio: %.3g' % acceptance_ratio)
        self.attempts['add'] = self.attempts.get('add',0) + 1
        if (np.random.rand() < acceptance_ratio) or force_acceptance:
            if force_acceptance:
                logger.warning('Forcing acceptance for debugging purposes.')
            comment += ': accepted (ratio %.3g)' % acceptance_ratio
            self.successes['add'] = self.successes.get('add',0) + 1
            # Choose which of the previously marginalized-out 
            # primitives we actually want to add at this position.
            k = choose_from_relative_probabilities(
                np.exp(candidate_distribution)
            )
            comment += ': adds detector %d' % k
            primitives = self.model.rule_population.flat_rules
            # print 'ADD DEBUG: ri %d pi %d k %d case %d' % (
            #     rule_index, primitive_index, k, debugcase
            # )
            # print 'ADD DEBUG: len nr %d' % len(new_rl)
            # print 'ADD DEBUG: len nr[ri] %d all lens %s' % (
            #     len(new_rl[rule_index]), 
            #     str(map(len,new_rl.rules)))
            # print 'cr all lens %s' % str(map(len,current_rl))
            new_rl[rule_index][primitive_index] = PrimitiveRule(*primitives[k])
            self.current_state.rl = new_rl
            self.ensure_current_rl_sorted()
            self.current_X = self.model.data.covariate_matrix(
                self.current_state.rl
            )
            # CAUTION: now beta is wrong
        else:
            comment += ': rejected (ratio %.3g)' % acceptance_ratio

        logger.info(comment)
        self.comments.append(comment)

    def remove(self, dry_run=False):
        current_rl = self.current_state.rl
        # Deal with the case where the list is empty.
        if len(current_rl) == 0:
            logger.info('Remove block: Nothing to remove.')
            self.comments.append('Nothing to remove')
            return
        
        # Choose which primitive to remove
        positions = [(i,j) for i,subrule in 
                     enumerate(current_rl.rules) for 
                     j,_ in enumerate(subrule)]
        rule_index, primitive_index = random.choice(positions)
        comment = 'remove detector %d from rule %d' % (primitive_index,
                                                        rule_index)
        shorter_rule = current_rl.copy()
        if len(current_rl[rule_index]) == 1:
            debugcase = 'a'
            consider_function = self.consider_addition_to_new_rule
            del shorter_rule[rule_index]
            copies_removed_primitive_in_rule = 1.0
        else:
            debugcase = 'b'
            consider_function = self.consider_addition_to_existing_rule
            copies_removed_primitive_in_rule = 0.0
            removed_primitive_as_int = (
                self.model.rule_population.get_primitive_index(
                    current_rl[rule_index][primitive_index]
                )
            )
            for primitive in current_rl[rule_index]:
                k = self.model.rule_population.get_primitive_index(primitive)
                if k == removed_primitive_as_int:
                    copies_removed_primitive_in_rule += 1.0
            del shorter_rule[rule_index][primitive_index]

        # print 'remove debug case %s %d,%d' % (
        #     debugcase, rule_index, primitive_index
        # )
        result = consider_function(
            shorter_rule, rule_index
        )
        # Note that we throw away the calculated marginalized
        # probability of proposing the removal step we have just
        # proposed-- what we want to use in calculating the acceptance
        # probability below is the _specific_ probability of proposing
        # this transition from our exact starting state.
        (_, relative_prior, 
         relative_likelihood, _, _) = result
        logger.debug('%s: rp %.3g, rf %.3g, sr %s' % (
            debugcase, relative_prior, 
            relative_likelihood, str(shorter_rule))
        )

        logger.debug('Removing detector %d from rule %d' % (primitive_index, rule_index))
        index = lambda r: self.model.rule_population.get_primitive_index(r)
        logger.debug('Rule is %s' % str(map(index, current_rl[rule_index])))
        logger.debug('copies of removed detector in rule: %d' % copies_removed_primitive_in_rule)
        logger.debug('Shorter RL is:')
        for line in str(shorter_rule).split('\n'):
            logger.debug(line)
        logger.debug('Raw relative prior %.3f' % relative_prior)

        # print shorter_rule
        # Invert the relative prior and likelihood, as we 
        # are _reversing_ the transition to S*
        relative_prior *= -1.0
        relative_likelihood *= -1.0

        probability_reverse_option = 1./(2.*len(shorter_rule) + 1.)
        probability_this_option = (copies_removed_primitive_in_rule /
                                   float(len(positions)))

        logger.debug('Removal block: %s' % comment)
        logger.debug('Prior, log(marg. new/old): %.3g' % 
                     relative_prior)
        logger.debug('Move choice prob (rev/for): %.3g/%.3g' % 
                     (probability_reverse_option, probability_this_option))
        logger.debug('Likelihood, log(marg. new/old): %.3g' % 
                     relative_likelihood)
        acceptance_ratio = (
            np.exp(relative_prior + relative_likelihood) *
            (probability_reverse_option / probability_this_option)
        )
        logger.debug('Acceptance ratio: %.3g' % acceptance_ratio)
        if dry_run:
            logger.debug('dry run flag set, aborting before accept check')
            return
        self.attempts['remove'] = self.attempts.get('remove',0) + 1
        if np.random.rand() < acceptance_ratio:
            comment += ': accepted (ratio %.3g)' % acceptance_ratio
            self.successes['remove'] = self.successes.get('add',0) + 1
            self.current_state.rl = shorter_rule
            shorter_X = self.model.data.covariate_matrix(shorter_rule)
            self.current_X = shorter_X
            # No need to ensure the current RL is sorted here;
            # possibly removing one entry from a sorted list preserves
            # its order.
            # CAUTION: beta is wrong
        else:
            comment += ': rejected (ratio %.3g)' % acceptance_ratio

        logger.info(comment)
        self.comments.append(comment)
        
    def move_primitive(self):
        """ Choose a primitive, sample over its possible positions in the list.

        Includes moving it to new rules; for each position, both the primitive
        and its inverse are considered.

        """ 
        base_rl = self.current_state.rl.copy()
        if len(base_rl) == 0 or (len(base_rl) == 1 and len(base_rl[0]) == 1):
            self.comments.append('Null detector move')
            logger.info('Null detector move')
            return

        # Choose primitive to consider moving. Each primitive has equal 
        # probability of being chosen.
        primitives = []
        for i, rule in enumerate(base_rl.rules):
            for j, primitive in enumerate(rule):
                primitives.append((i,j,primitive))
        index = np.random.choice(len(primitives))
        i,j,primitive = primitives[index]
        
        # Document the choice, and prepare a reversed version of the 
        # primitive 
        comment_prefix = 'detector (%d,%d) moves to ' % (i,j)
        inverted_primitive = PrimitiveRule(*primitive.as_tuple())
        if primitive.direction == 'above':
            inverted_primitive.direction = 'below'
        else:
            inverted_primitive.direction = 'above'
        logger.debug('move_primitive: forward %s %s, reverse %s %s' %
                    (primitive, primitive.direction, inverted_primitive, 
                     inverted_primitive.direction))

        # Delete the original copy of the primitive (and its
        # containing rule if necessary.)  Following this deletion,
        # base_rl is R' from section 5.4.
        del base_rl[i][j]
        if len(base_rl[i]) == 0:
            del base_rl[i]

        # Collect all the possible rule lists we can form by inserting
        # the primitive of interest in existing or new rules. (This is 
        # the set \Delta in the notation of section 5.4.) Each candidate
        # is accompanied by a comment describing the process that generated
        # it. 
        comments = []
        rls = []
        # First the cases where we insert the primitive in an existing rule
        # (\Delta_2^+ and \Delta_2^- in section 5.4; note here we 
        # intermingle them rather than constructing them separately.) 
        for i, rule in enumerate(base_rl.rules):
            for p, comment in [(primitive, 'existing rule %d'),
                               (inverted_primitive, 
                                'existing rule %d (inverted)')]:
                rl = base_rl.copy()
                rl[i].append(p)
                comments.append(comment % i)
                rls.append(rl)
        # Then the cases where we insert the primitive as a new rule
        # (\Delta_1^+ and \Delta_1^- in section 5.4.) 
        for i in xrange(len(base_rl)+1):
            for p, comment in [(primitive, 'new rule at position %d'),
                               (inverted_primitive, 
                                'new rule at position %d (inverted)')]:
                rl = base_rl.copy()
                rl.rules = rl.rules[:i] + [[p]] + rl.rules[i:]
                comments.append(comment % i)
                rls.append(rl)

        # Now evaluate the posterior distribution over \Delta:
        priors = np.zeros(len(rls))
        likelihoods = np.zeros(len(rls))
        for k,rl in enumerate(rls):
            # We could do this more efficiently, because we have a lot
            # of information about the structure of each candidate
            # rule list-- for example, we know that the prior
            # contribution coming from the choice of primitives has
            # not changed, and for each candidate rule list, the
            # multinomial term for at most two composite rules has
            # changed relative to base_rl-- but the set \Delta is
            # small compared to the total number of primitives (at
            # least so we typically expect!)  so we take the less
            # efficient route for clarity and readability at a limited
            # speed cost.
            # 
            # We need to supply a full state to the prior function,
            # because the prior depends on the values of eg the
            # phylogenetic prior hyperparameters
            candidate_state = self.current_state.copy()
            candidate_state.rl = rl 
            priors[k] = self.model.rule_list_prior(candidate_state)
            # This function calculates only the prior contributions
            # from the rule list, which is fine, as nothing else is
            # being varied between candidates here.
            new_X = self.model.data.covariate_matrix(rl)
            likelihoods[k] = self.model.likelihood_z_given_X(
                self.current_state.omega,
                new_X
            )
        # Summarize the results in debugging output.
        logger.debug('Move options:')
        for c,p,l,rl in zip(comments, priors, likelihoods, rls):
            logger.debug(c)
            logger.debug('%.3g' % p)
            logger.debug('%.3g' % l)
            logger.debug('%s' % rl)
            
        # Choose from among the options according to the posterior 
        # probabilities, and excecute and record the choice.
        k = choose_from_relative_probabilities(np.exp(priors + likelihoods))
        self.current_state.rl = rls[k]
        self.ensure_current_rl_sorted()
        self.current_X = self.model.data.covariate_matrix(
            self.current_state.rl
        )
        # CAUTION: beta is wrong.
        self.comments.append(comment_prefix + comments[k])
        logger.info(comment_prefix + comments[k])

##### END CORE SAMPLING CODE
########################################
########################################
########################################


########################################
# UTILITIES

def probabilities(rule_list, beta, test_data):
    """ 
    Predict probabilities of outcome according to the rule list.

    """
    X = test_data.covariate_matrix(rule_list)
    psi = np.dot(X,beta)
    probabilities = 1.0/(1.0+np.exp(-1.0*psi))
    return probabilities

def prediction(rule_list, beta, test_data):
    """ 
    Predict y=1 for subjects with probability >= 0.5.

    """
    X = test_data.covariate_matrix(rule_list)
    psi = np.dot(X,beta)
    probabilities = 1.0/(1.0+np.exp(-1.0*psi))
    prediction = probabilities >= 0.5
    return prediction


class AddRemoveOnlyPriorSampler(LogisticRuleSampler):
    def step(self):
        move_type_indicator = np.random.rand()
        if move_type_indicator < -1.0:# 0.44
            self.update_primitives()
        elif move_type_indicator < 2.0:#0.66
            self.add_remove()
        else:
            raise
#            self.move_primitive()
        # If the structure moves are accepted, beta becomes out of date, 
        # perhaps grievously, so we defensively update it before
        # recording the new state
        self.states.append(self.current_state.copy())



class AddRemoveUpdatePriorSampler(LogisticRuleSampler):
    def step(self):
        move_type_indicator = np.random.rand()
        if move_type_indicator < 0.5:# 0.44
            self.update_primitives()
        elif move_type_indicator < 2.0:#0.66
            self.add_remove()
        else:
            raise
#            self.move_primitive()
        # If the structure moves are accepted, beta becomes out of date, 
        # perhaps grievously, so we defensively update it before
        # recording the new state
        self.states.append(self.current_state.copy())

class UniformPhylogenyModel(LogisticRuleModel):

    def flat_prior(self, state):
        """ Evaluate log-probability of each primitive in the population.

        Return value is properly normalized.
        
        This subclass ignores phylogenetic weights.
        
        """
        total_duration = float(self.data.experiment_end - self.data.experiment_start)
        durations = (self.rule_population.flat_durations /
                     ((1.0+self.window_duration_epsilon)*total_duration)
                    )
        window_a = (
            state.window_concentration *
            state.window_typical_fraction
        )
        window_b = (
            state.window_concentration *
            (1.0-state.window_typical_fraction)
        )
        window_weights = scipy.stats.beta.logpdf(
            durations, 
            window_a,
            window_b
        )

        weights = window_weights
        normalization = logsumexp(weights)
        return weights - normalization



