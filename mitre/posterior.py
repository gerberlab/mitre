"""
Utilities for summarizing the posterior distribution of logistic rule models.

Also includes output for mixing diagnostics, which aren't summaries of
the posterior as such.

"""
import logit_rules 
import rules 
import prior_by_primitive as pbp
import numpy as np
import scipy.stats
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.special import gammaln
from scipy.misc import logsumexp
import matplotlib.pyplot as plt

import logging
logger = rules.logger

def log_multinomial_coefficient(list_of_tuples):
    counts = dict.fromkeys(set(list_of_tuples),0)
    for t in list_of_tuples:
        counts[t] += 1
    N = len(list_of_tuples)
    v = np.array(counts.values())
    return gammaln(N+1)-np.sum(gammaln(v+1))


####
# Parameters (this should be made user-configurable later)
N_best = 5 # show the best N_best of everything

class PosteriorSummary():
    """ Collect information about the posterior distribution. """
    
    def __init__(self, sampler, burnin_fraction=0.1, tag=None, primitive_prior=None):
        """ Create an object to calculate/store information about sampled posterior.

        Arguments:
        sampler - a LogisticRuleSampler instance
        burnin_fraction - what fraction of samples to discard (float, default 0.1)
        tag - prefix for output filenames (string or None (the default), in which case no
           prefix is used) (not implemented yet)
        primitive_prior - dict mapping primitive rules (as tuples) to their prior 
           probability, marginalizing out hyperparameters, or None (the default), in which
           case they will be recalculated as needed (in which case they will be saved, in 
           pickled form, for reuse.)

        """
        self.sampler = sampler
        self.model = sampler.model
        if tag is None:
            self.tag = ''
        else:
            self.tag = tag + '_'
        self.primitive_prior = primitive_prior
 
        # Select only the samples we wish to keep, collect them in attributes of self;
        # tabulate 

        self.burnin = int(burnin_fraction * len(sampler.states))
        # There is an off-by-one problem comparing the states to the beta
        # samples: each iteration first updates beta many times and records
        # the values, then updates the structure once, updates beta once, and
        # records the state. Work around this:
        self.state_ensemble = sampler.states[int(self.burnin):-1]
        self.ensemble_beta_samples = sampler.additional_beta_values[int(self.burnin)+1:]
        # We also want to track the priors and likelihoods associated with the states 
        # in the ensemble:
        self.ensemble_likelihoods = np.array(sampler.likelihoods[int(self.burnin):-1])
        self.ensemble_priors = np.array(sampler.priors[int(self.burnin):-1])
        logger.info('Tabulating posterior distributions...')
        self.tabulate()

        # Collect reports which will normally be saved to files, so that there's
        # an easy interactive way to get at them.
        self.reports = {}

    def mixing_diagnostics(self):
        """ Plot various parameters versus MCMC iteration, for diagnostic purposes.

        Values output include the likelihood, prior, rule list length
        (total number of primitives,) a subset of the auxiliary
        variables omega, and the parameters phylogeny_mean, phylogeny_std,
        window_concentration, window_typical_fraction.

        Plots are saved as PDF figures.

        """
        n_subjects = self.model.data.n_subjects
        sampler = self.sampler
        if n_subjects < 50:
            omega_spacing = 5
        else:
            omega_spacing = 20
        omegas = np.vstack([state.omega for state in sampler.states])[:,::n_subjects]

        vectors = [sampler.likelihoods, 
                   sampler.priors,
                   [sum(map(len,state.rl)) for state in sampler.states],
                   omegas,]
        titles = ['Likelihood (log)',
                  'Prior (log)',
                  'Length (total number of primitives)',
                  'Auxiliary variables omega (subset of subjects)']
        labels = ['likelihood',
                  'prior',
                  'lengths',
                  'omegas']

        for v in ['phylogeny_mean', 'phylogeny_std', 
                  'window_concentration', 'window_typical_fraction']:
            array = [getattr(state, v) for state in sampler.states]
            vectors.append(array)
            titles.append(v)
            labels.append(v)

        for vector, title, label in zip(vectors, titles, labels):
            plt.figure()
            plt.plot(vector)
            plt.title(title)
            plt.xlabel('MCMC iteration')
            plt.savefig('%s%s.pdf' % (self.tag, label))

    def tabulate(self):
        """ Count samples primitives appear in, make posterior distributions, collect other info.
        
        Also determines the length distribution and the typical
        length, subrule length distribution and typical subrule
        length, distribution of total number of primitives in each RL
        and typical number of primitives, collections of beta
        coefficients sorted by state, subrule and primitive, and the
        values of the phylogeny and window hyperparameters, (we are
        looping through all the samples anyway, so we might as well.)

        """
        self.n_states = 0
        self.n_subrules = 0
        self.beta_by_state = {}
        # Beta by primitive is a weird thing to track as of course the
        # individual coefficients apply to the subrule, and not the
        # primitive, truth; but it is certainly useful to know, e.g.,
        # does this primitive typically appear in rules with positive
        # coefficients, or negative ones?
        self.beta_by_primitive = {}
        self.beta_by_subrule = {}
        self.indices_by_state = {}
        self.indices_by_subrule = {}
        self.indices_by_primitive = {}
        self.constant_term_beta = [] # not sure what we will do with this information, but might as well
        self.rl_lengths = []
        self.rl_total_primitives = []
        self.subrule_lengths = []
        self.state_ensemble_primitives = []
        hyperparameters = ['phylogeny_mean',
                           'phylogeny_std',
                           'window_concentration',
                           'window_typical_fraction',]
        self.hyperparameter_samples = {p: [] for p in hyperparameters}
        for state_index, (state, beta_samples) in enumerate(
            zip(self.state_ensemble,self.ensemble_beta_samples)
            ):
            for p in hyperparameters:
                self.hyperparameter_samples[p].append(
                    getattr(state, p)
                )
            beta = np.vstack(beta_samples)
            rl = state.rl.as_tuple()
            self.constant_term_beta.append(beta[-1])
            self.rl_lengths.append(len(rl))
            self.rl_total_primitives.append(sum(map(len,rl)))

            self.n_states += 1
            self.beta_by_state.setdefault(rl, []).append(beta)
            self.indices_by_state.setdefault(rl,[]).append(state_index)
            this_state_primitives = set()
            for i,subrule in enumerate(rl):
                self.n_subrules += 1
                self.subrule_lengths.append(len(subrule))
                self.beta_by_subrule.setdefault(subrule, []).append(beta[:,i])
                self.indices_by_subrule.setdefault(subrule, []).append(state_index)
                primitives_in_subrule = set()
                for j,primitive in enumerate(subrule):
                    primitives_in_subrule.add(primitive)
                    self.beta_by_primitive.setdefault(primitive, []).append(beta[:,i])
                this_state_primitives.update(primitives_in_subrule)
            for primitive in this_state_primitives:
                self.indices_by_primitive.setdefault(primitive, []).append(state_index)
            self.state_ensemble_primitives.append(this_state_primitives)

        # It is possible there are no subrules, particularly in debugging cases.
        # In this case treat the effective maximum length as 0
        if len(self.subrule_lengths) == 0:
            self.max_subrule_length = 0
        else:
            self.max_subrule_length = max(self.subrule_lengths)

        for d in [self.beta_by_primitive, self.beta_by_subrule]:
            for k in d:
                d[k] = np.hstack(d[k])
        # beta_by_state needs to be formatted differently, to preserve information
        # about which coefficients correspond to which rules.
        for k in self.beta_by_state:
            self.beta_by_state[k] = np.vstack(self.beta_by_state[k])

        # Normalize the index counts to obtain posterior distributions
        self.posterior_by_primitive = {}
        # For the posterior we need iterate only over the primitives we have seen in the samples
        for primitive, indices in self.indices_by_primitive.iteritems():
            self.posterior_by_primitive[primitive] = len(indices)

        # posterior_by_primitive[k] is the posterior probability, estimated, that at least
        # one copy of primitive k appears in the true rule list
        normalization = float(len(self.state_ensemble))
        for k in self.posterior_by_primitive:
            self.posterior_by_primitive[k] = self.posterior_by_primitive[k] / normalization

        self.posterior_by_subrule = {}
        for subrule, indices in self.indices_by_subrule.iteritems():
            self.posterior_by_subrule[subrule] = len(indices)/float(self.n_subrules)
        self.posterior_by_state = {}
        for state, indices in self.indices_by_state.iteritems():
            self.posterior_by_state[state] = len(indices)/float(self.n_states)
            
        for l in ['rl_lengths', 'rl_total_primitives', 'subrule_lengths']:
            setattr(self, l, np.array(getattr(self, l)))
        self.rl_length_distribution = {
            d: sum(self.rl_lengths == d) for d in xrange(max(self.rl_lengths)+1)
        }
        self.subrule_length_distribution = {
            d: sum(self.subrule_lengths == d) for d in xrange(self.max_subrule_length+1)
        }
        self.rl_total_primitive_distribution = {
            d: sum(self.rl_total_primitives == d) for d in 
            xrange(max(self.rl_total_primitives)+1)
        }
        # This is lazy, but okay...
        self.normalized_rl_total_primitive_distribution = {
            d: sum(self.rl_total_primitives == d)/float(self.n_states) for d in 
            xrange(max(self.rl_total_primitives)+1)
        }
        self.rl_length_posterior = [(v/float(self.n_states),k) for k,v in
                                    self.rl_length_distribution.iteritems()]
        self.rl_length_posterior.sort(reverse=True)
        self.rl_total_primitive_posterior = [(v/float(self.n_states),k) for k,v in
                                    self.rl_total_primitive_distribution.iteritems()]
        self.rl_total_primitive_posterior.sort(reverse=True)
        self.modal_rl_length = self.rl_length_posterior[0][1]
        self.modal_rl_total_primitives = self.rl_total_primitive_posterior[0][1]

        if (self.modal_rl_length == 0) and len(self.rl_length_posterior) > 1:
            self.modal_nonzero_rl_length = self.rl_length_posterior[1][1]
        else:
            self.modal_nonzero_rl_length = self.modal_rl_length

        if ((self.modal_rl_total_primitives == 0) and 
            len(self.rl_total_primitive_posterior) > 1):
            self.modal_nonzero_rl_total_primitives = self.rl_total_primitive_posterior[1][1]
        else:
            self.modal_nonzero_rl_total_primitives = self.modal_rl_total_primitives


    ########################################
    # UTILITY
    def format_rl(self, rule_list, beta=None):
        """ Explain a rule list, in the context of the model.
        
        Intended to be prettier and more flexible than model.show();
        in particular, we can incorporate a distribution of coefficients.
        Returns a string.

        Arguments:
        rule_list - rule list, either as a list of lists of PrimitiveRule objects
        or a list of lists of tuples representing primitives.

        beta - None, the default, or a vector of length len(rl) +
        self.model.data.n_fixed_covariates, or an array with number of
        rows equal to len(rl) + self.model.data.n_fixed_covariates,
        each row of which is one sample from the distribution over
        vectors beta (of length len(state.rl) +
        self.model.data.n_fixed_covariates). In the array case, median
        and 95% CI will be displayed for each coefficient, and the
        median will be used in calculating the likelihood.

        """
        model = self.model
        if beta is not None and len(beta.shape) > 1:
            beta_distribution = beta
            beta = np.median(beta, axis=0)
            beta_low, beta_high = np.percentile(beta_distribution, [2.5, 97.5], axis=0)
        else:
            beta_distribution = None
        # If we have been given a list of tuples, convert it into a
        # list of PrimitiveRule objects
        if len(rule_list) > 0:
            if isinstance(rule_list[0][0], tuple):
                rule_list = logit_rules.RuleList(rule_list,model=self.model)

        X = self.model.data.covariate_matrix(rule_list)
        if beta is None:
            total_likelihood = 'n/a'
        else:
            total_likelihood = '%.3g' % self.model.likelihood_y_given_X_beta(X,beta)
        N_total = self.model.data.n_subjects
        likelihood_line = 'overall likelihood %s' % total_likelihood
        header_line = 'Rule list with %d rules (%s):' % (len(rule_list), likelihood_line)
        base_rate_line = ('Frequency of positive outcome in dataset: %.3f (%d/%d)' % 
                          (sum(self.model.data.y)/float(N_total),
                           sum(self.model.data.y),N_total))
        overall_lines = [header_line]

        logit = lambda t: (1.0/(1.0+np.exp(-1.0*t)))        
        constant_term_index = len(rule_list) # ie, the first coefficient after those for the rules
        if beta is None: 
            coefficient = lambda i: 'n/a'
            effect = lambda i: 'n/a'
            default_probability = 'n/a'
        elif beta_distribution is None:
            coefficient = lambda i: '%.3g' % beta[i]
            effect = lambda i: ('Odds of positive outcome INCREASE by factor of %.3g' % np.exp(beta[i]) if
                                beta[i] > 0 else
                                'Odds of positive outcome DECREASE by factor of %.3g' % np.exp(-1.0*beta[i]))
            default_probability = '%.3g' % logit(beta[constant_term_index])
        else:
            coefficient = lambda i: '%.3g (%.3g -- %.3g)' % (beta[i], beta_low[i], beta_high[i])
            default_probability = '%.3g (%.3g -- %.3g)' % tuple(map(logit,
                                                                    (beta[constant_term_index], 
                                                                     beta_low[constant_term_index], 
                                                                     beta_high[constant_term_index])))
            effect = lambda i: ('Odds of positive outcome INCREASE by factor of %.3g (%.3g - %.3g)' % 
                                (np.exp(beta[i]), np.exp(beta_low[i]), np.exp(beta_high[i])) if
                                beta[i] > 0 else
                                'Odds of positive outcome DECREASE by factor of %.3g (%.3g - %.3g)' % 
                                (np.exp(-1.0*beta[i]), np.exp(-1.0*beta_high[i]), np.exp(-1.0*beta_low[i])))
            # Note the high-low swap here is deliberate: we are effectively resorting by absolute
            # value.

        # initializing i here lets us smoothly increment it later
        i = -1
        for i,rule in enumerate(rule_list):
            rule_lines = []
            for p in rule:
                line = ('Between time %.3f and time %.3f, variable %s %s is %s %.4f' %
                        (p.window[0],p.window[1],
                         self.model.data.variable_names[p.variable],
                         p.type_,p.direction,p.threshold))
                rule_lines.append('\t\t' + line)
            N = np.sum(X[:,i])
            k = np.sum(self.model.data.y[X[:,i]>0.])
            this_coefficient = coefficient(i)
            this_effect = effect(i)
            application_line = ('This rule applies to %d/%d (%.3f) subjects in dataset, '
                                '%d/%d with positive outcomes (%.3f).' %
                                 (N,N_total,N/float(N_total),
                                  k,N,k/float(N)))
            overall_lines.append('\nRule %d (coefficient %s):\n\t %s, if:' % 
                                 (i,this_coefficient,this_effect)
                                )
            overall_lines = overall_lines + rule_lines 
            overall_lines.append(application_line)

        # Describe the constant rule
        i += 1
        constant_term_line = ('\nConstant term (coefficient %s):\n'
                              '\tPositive outcome probability %s if no other rules apply'
                              % (coefficient(i),default_probability))

        overall_lines.append('\n')

        for feature, value in self.model.data.additional_covariate_encoding:
            i += 1
            this_coefficient = coefficient(i)
            this_effect = effect(i)
            covariate_line = ('Effect of feature "%s" == "%s" (coefficient %s): %s' %
                              (feature, value, this_coefficient, this_effect))
            overall_lines.append(covariate_line)

        overall_lines.append(constant_term_line)

        return '\n'.join(overall_lines) + '\n'


    ########################################
    # HIGH-LEVEL SUMMARY
    
    def _quick_report(self):
        # Length distribution, point summary, point and ensemble accuracy.
        self.point_summarize()
        self.make_point_prediction()
        self.make_ensemble_prediction()
        lines = []
        lines.append('POINT SUMMARY:')
        lines.append(self.format_rl(self.point.rl, self.point_betas))
        lines.append('POINT SUMMARY CLASSIFIER PERFORMANCE:')
        lines.append(
            self.classifier_accuracy_report(
                self.point_probabilities()
                )
        )
        lines.append('ENSEMBLE CLASSIFIER PERFORMANCE:')
        lines.append(
            self.classifier_accuracy_report(
                self.ensemble_probabilities()
                )
        )
        lines.append('%d total samples after %d burnin' %
                     (len(self.state_ensemble), self.burnin))
        lines.append('Most frequent rule list lengths (counting total number of primitives):')
        lines.append('\tfrequency\tlength')
        for frequency, n in self.rl_total_primitive_posterior[:N_best]:
            lines.append('\t%.3g\t\t%d' % (frequency, n))

        prior_empty_odds = self.model.hyperparameter_a_empty / (self.model.hyperparameter_b_empty)
        posterior_empty_odds = (self.normalized_rl_total_primitive_distribution[0] /
                                (1.0 - self.normalized_rl_total_primitive_distribution[0]))
        bf_empty = posterior_empty_odds / prior_empty_odds
        
        lines.append('\n Bayes factor for the empty rule set: %.3g' % bf_empty)

        report = '\n'.join(lines) + '\n'
        self.reports['quick'] = report
        return report

    def quick_report(self):
        print self._quick_report()

    def all_summaries(self, n_clusters=100):
        # Run all the standard summaries, and then synopsize their results.
        quick = self._quick_report()
        self.do_clustering()
        table = self.cluster_table(n_clusters)
        report = (quick + 
                  '\n Best (highest posterior probability) rule list clusters: \n' +
                  '\n'.join(table) + '\n')
        self.reports['full'] = report
        return report

    def bayes_summary(self, N_primitive_prior_samples=10000):
        if self.primitive_prior is None:
            self.calculate_primitive_prior(N_primitive_prior_samples)
        self.calculate_higher_level_priors_and_posteriors()
        self.calculate_bayes_factors()
        return self.write_bayes_factor_table()

    def point_summarize(self):
        """ Chooses a point summary of the posterior, sets self.point.
        
        No return or output.

        The point summary is the sampled state with highest posterior
        probability, among all those states with the modal length
        (measured in total primitives,) unless the modal length is
        zero and the posterior probability of the empty list is < 0.5,
        in which cae we use the modal nonzero length.

        Also sets self.point_betas to the corresponding 
        entry from additional_beta_samples.

        """
        if self.normalized_rl_total_primitive_distribution[0] < 0.5:
            target_length = self.modal_nonzero_rl_total_primitives
        else:
            target_length = self.modal_rl_total_primitives

        states = []
        posteriors = []
        betas = []
        for (state, prior, likelihood, beta) in zip(self.state_ensemble,
                                                    self.ensemble_priors,
                                                    self.ensemble_likelihoods,
                                                    self.ensemble_beta_samples):
            total_primitives = sum(map(len,state.rl))
            if total_primitives == target_length:
                states.append(state)
                posteriors.append(prior+likelihood)
                betas.append(beta)
        best_index = np.argmax(posteriors)
        self.point = states[best_index]
        self.point_betas = np.vstack(betas[best_index])

    def make_point_prediction(self, test_data=None):
        """ Uses the point summary to make binary predictions for each subject.

        Must be run after self.point_summary. 

        If test_data is None, the default, self.model.data will be
        used. In this case, this method sets self.ensemble_prediction.

        Returns: vector of predicted outcomes for each subject in 
        test_data.

        """
        save_prediction = False
        if test_data is None:
            test_data = self.model.data
            save_prediction = True
        prediction = logit_rules.prediction(
            self.point.rl,
            self.point.beta,
            test_data=test_data
        )
        if save_prediction:
            self.point_prediction = prediction
        return prediction


    def point_probabilities(self, test_data=None):
        """ Uses the point summary to predict outcome probabilities for each subject.

        Must be run after self.point_summary. 

        If test_data is None, the default, self.model.data will be
        used.

        Returns: vector of predicted positive outcome probabilities
        for each subject in test_data.

        """
        if test_data is None:
            test_data = self.model.data
        return logit_rules.probabilities(
            self.point.rl,
            self.point.beta,
            test_data=test_data
        )

    def make_ensemble_prediction(self, test_data=None):
        """ Uses the state ensemble to make binary predictions for each subject. 
        
        A subject is assigned predicted outcome 1 iff a majority of
        the states in the ensemble predict that outcome.

        If test_data is None, the default, self.model.data will be
        used. In this case, this method sets self.ensemble_prediction.

        Returns: vector of predicted outcomes for each subject in 
        test_data.

        """
        save_prediction = False
        if test_data is None:
            test_data = self.model.data
            save_prediction = True
        ensemble_all_predictions = np.vstack(
            [logit_rules.prediction(state.rl,
                                    state.beta,
                                    test_data=test_data) for
             state in self.state_ensemble]
        )
        ensemble_vote_fraction = (np.sum(ensemble_all_predictions,axis=0) / 
                                  float(len(self.state_ensemble)))
        prediction = ensemble_vote_fraction >= 0.5
        if save_prediction:
            self.ensemble_prediction = prediction
        return prediction

    def ensemble_probabilities(self, test_data=None):
        """ Uses the state ensemble to predict outcome probabilities for each subject.
        
        The predicted probability is the average of the probabilities
        predicted by each state in the ensemble.

        If test_data is None, the default, self.model.data will be
        used.

        Returns: vector of predicted positive outcome probabilities
        for each subject in test_data.

        """
        if test_data is None:
            test_data = self.model.data
        ensemble_all_predictions = np.vstack(
            [logit_rules.probabilities(state.rl,
                                       state.beta,
                                       test_data=test_data) for
             state in self.state_ensemble]
        )
        ensemble_mean_probability = (np.sum(ensemble_all_predictions,axis=0) / 
                                     float(len(self.state_ensemble)))
        return ensemble_mean_probability

    def classifier_accuracy_report(self, prediction_vector, threshold=0.5):
        """ Determine AUC and other metrics, write report.

        prediction_vector: vector of booleans (or outcome
        probabilities) of length n_subjects,
        e.g. self.point_predictions, self.ensemble_probabilities()...
        If this has dtype other than bool, prediction_vector > threshold
        is used for the confusion matrix.

        Returns: one string (multiple lines joined with \n, including
        trailing newline) containing a formatted report.

        """
        auc = roc_auc_score(self.model.data.y.astype(float), prediction_vector.astype(float))
        if not (prediction_vector.dtype == np.bool):
            prediction_vector = prediction_vector >= threshold
        conf = confusion_matrix(self.model.data.y, prediction_vector)
        
        lines = ['AUC: %.3f' % auc,
                 'Confusion matrix: \n\t%s' % str(conf).replace('\n','\n\t')]
        return '\n'.join(lines) + '\n'


    ######################################## 
    # BAYES-FACTOR-BASED METHODS

    def calculate_primitive_prior(self, N_samples):
        logger.info('Drawing samples from prior distribution for Bayes factor calculations...')
        self.base_prior_by_primitive = pbp.estimate_primitive_priors_lognormal(self.model, N_samples)
        self.length_prior = pbp.estimate_length_distribution(self.model, 10*N_samples)

    def prior_and_posterior_for_category(self, category_to_primitives):
        category_to_prior = {}
        category_to_posterior = {}
        # We want to take the logs of the lengh prior, but some elements 
        # will be zero; entries in the result being -inf is okay, but we want
        # to avoid a potentially alarming warning message
        M = np.zeros(self.length_prior.shape)
        M[self.length_prior > 0.] = np.log(self.length_prior[self.length_prior > 0.])
        M[self.length_prior <= 0.] = -np.inf
        lengths = np.arange(len(M))

        for category, primitives in category_to_primitives.iteritems():
            base_prior = 0.
            for primitive in primitives:
                base_prior += self.base_prior_by_primitive[primitive]
            # Effectively we calculate the probability of never
            # choosing this category in a rule with 0 primitives, 1
            # primitive, etc, then add those up, scaling by the
            # probability of having 0 primitives, 1 primitive, etc.;
            # for speed and numerical behavior we do this in a
            # vectorized way on a log scale:
            inclusion_prior = -1.0*np.expm1(
                logsumexp(M + (np.log1p(-1.0*base_prior) * lengths))
            )
            category_to_prior[category] = inclusion_prior

            states_in_category = 0
            for state in self.state_ensemble_primitives:
                if state.intersection(primitives):
                    states_in_category += 1
            category_to_posterior[category] = (
                states_in_category / float(len(self.state_ensemble))
            )
        return category_to_prior, category_to_posterior


    def calculate_higher_level_priors_and_posteriors(self):
        # self.base_prior_by_primitive is the prior probability of
        # choosing each primitive to fill an empty slot; we want the
        # prior probability that each primitive appears at least once
        # in the rule list. We calculate this in the same way as we do
        # prior probabilities of seeing at least one primitive in
        # various other categories, e.g., those that depend on a
        # particular variable, so we'll set them all up at once:
        logger.info('Higher level tabulation begins')
        primitive_to_self = {}
        # We use posterior_by_primitive here as a convenient list 
        # of those primitives which appear at least once in the samples.
        for primitive in self.posterior_by_primitive:
            primitive_to_self[primitive] = {primitive}

        variable_to_primitives = {}
        for primitive in self.base_prior_by_primitive:
            variable_name = self.model.data.variable_names[primitive[0]]
            variable_to_primitives.setdefault(variable_name,set()).add(primitive)

        # For visualization purposes, set up a fairly tricky one.
        #
        # We define a grid whose rows are the atomic time intervals
        # and whose columns are the leaf nodes of the tree of
        # variables. We associate a probability with each cell of the
        # grid, specifically, the probability of observing any
        # primitive whose time window includes the corresponding
        # atomic window and whose variable is that leaf variable or
        # any of its ancestors.
        grid_cell_to_primitives = {}
        interval_endpoints = np.linspace(self.model.data.experiment_start,
                                         self.model.data.experiment_end,
                                         self.model.rule_population.N_intervals + 1)
        interval_midpoints = 0.5*(interval_endpoints[:-1] + interval_endpoints[1:])
        leaf_names = self.model.data.variable_tree.get_leaf_names()
        variable_name_to_leaf_names = {
            n.name: n.get_leaf_names() for n in
            self.model.data.variable_tree.traverse()
        }
        for primitive in self.base_prior_by_primitive:
            window_start, window_stop = primitive[1]
            variable_name = self.model.data.variable_names[primitive[0]]
            for timepoint in interval_midpoints:
                if window_start > timepoint or window_stop < timepoint:
                    continue
                for leaf in variable_name_to_leaf_names[variable_name]:
                    key = (timepoint, leaf)
                    grid_cell_to_primitives.setdefault(key,set()).add(primitive
           )

        l = [('primitive_inclusion_prior',
              'primitive_inclusion_posterior',
              primitive_to_self),
             ('variable_inclusion_prior',
              'variable_inclusion_posterior',
              variable_to_primitives),
             ('grid_inclusion_prior',
              'grid_inclusion_posterior',
              grid_cell_to_primitives)]
        for (prior_name, posterior_name, categories) in l:
            pr, pos = self.prior_and_posterior_for_category(
                categories
            )
            setattr(self, prior_name, pr)
            setattr(self, posterior_name, pos)

        # then use them in the next method, then also calculate a
        # Bayes factor for emptiness, then update the reports, then
        # launch

    def calculate_bayes_factors(self):
        self.bayes_factors_by_variable = {}
        self.bayes_factors_by_primitive = {}
        self.bayes_factors_by_grid_cell = {}
        warn_for_bad_priors = False
        warn_for_bad_posteriors = False
        max_good_posterior = np.float_(1.0) - np.finfo(np.float_).resolution
        for all_bfs, all_priors, all_posteriors in [
            (self.bayes_factors_by_variable, 
             self.variable_inclusion_prior,
             self.variable_inclusion_posterior),
            (self.bayes_factors_by_primitive, 
             self.primitive_inclusion_prior,
             self.primitive_inclusion_posterior),
            (self.bayes_factors_by_grid_cell, 
             self.grid_inclusion_prior,
             self.grid_inclusion_posterior),
            ]:
            for k, posterior in all_posteriors.iteritems():
                prior = all_priors[k]
                if prior == 0.:
                    warn_for_bad_priors = True
                    bf = np.inf
                elif posterior > max_good_posterior:
                    warn_for_bad_posteriors = True
                    bf = np.inf
                else:
                    bf = (posterior/prior) * ((1-prior) / (1-posterior))
                all_bfs[k] = bf
        if warn_for_bad_priors:
            logger.warning('Some prior probabilities were exactly zero; sampling '
                           'from the prior was probably not adequate.')
        if warn_for_bad_posteriors:
            logger.warning('Some posterior probabilities were equal to 1.0; sampling '
                           'from the posterior was probably not adequate.')
        # Now the BF for emptiness
        emptiness_prior = self.length_prior[0]
        emptiness_posterior = (
            self.rl_length_distribution[0] /
            float(sum(self.rl_length_distribution.values()))
        )
        emptiness_bf = (
            (emptiness_posterior/emptiness_prior) * 
            ((1-emptiness_prior) / (1-emptiness_posterior))
        )
        self.bayes_factor_emptiness = emptiness_bf

    def write_bayes_factor_table(self, N_to_print=10):
        identity = lambda x: x
        rename_variable = lambda i: self.model.data.variable_names[i]
        rename_primitive = lambda t: tuple([rename_variable(t[0])] +
                                           list(t)[1:])

        to_report = [
            ('Best-supported variables', self.bayes_factors_by_variable,
             self.variable_inclusion_posterior, len(self.state_ensemble),
             None, identity,),
            ('Best-supported detectors', self.bayes_factors_by_primitive,
             self.primitive_inclusion_posterior, len(self.state_ensemble),
             self.beta_by_primitive, rename_primitive),
            ]
        report = ['Bayes factor for empty rule: %.3g\n' %
                  self.bayes_factor_emptiness]
        for descriptor, bayes_factors, posterior, posterior_normalization, betas, mapper in to_report:
            report.append(descriptor+ ':\n')
            l = [(bf, item) for item, bf in bayes_factors.iteritems()]
            l.sort(reverse=True)
            l = l[:(len(l) if N_to_print is None else N_to_print)]
            metadata_lines = [] 
            item_lines = []
            for bf, item in l:
                metadata_line = (['Bayes factor: %.1g' % bf, 
                                   'N_samples: %d' %
                                  (posterior_normalization *
                                   posterior[item])])
                if betas:
                    median_beta = np.median(betas[item])
                    ci_beta = np.percentile(betas[item],[2.5,97.5])
                    metadata_line = (
                        metadata_line +
                        ['Beta (median): %.3g ' % median_beta, 
                         'Beta 95% CI:' + '(%.3g, %.3g)' % tuple(ci_beta)] 
                    )
                metadata_lines.append(metadata_line)
                item_lines.append('\t\t' + str(mapper(item)))
            metadata_lines = ['\t'+'\t'.join(line) for
                              line in metadata_lines]
            for description, item in zip(metadata_lines, item_lines):
                report.append(description)
                report.append(item)
                report.append('\n')

            report.append('\n')
        return '\n'.join(report)
    
    def visualize_bayes_factors(self):
        pass

    ########################################
    # CLUSTERING
    
    def cluster_primitives(self):
        test = primitive_can_join_cluster
        result = generic_cluster_from_posterior(self.posterior_by_primitive, test)
        self.primitive_clusters, self.primitive_cluster_posteriors = result
        self.primitives_to_clusters = {}
        for i,l in enumerate(self.primitive_clusters):
            for primitive in l:
                self.primitives_to_clusters[primitive] = i

    def subrule_to_cluster_form(self, subrule):
        """ Describe a subrule using the clusters its primitives belong to.

        Argument is a subrule expressed as a list of tuples.

        """
        return sorted([self.primitives_to_clusters[p] for p in subrule])

    def rule_pair_ok(self, subrule1, subrule2):
        """ Determine whether two rules are equivalent.

        Rules should be expressed as a list of tuples.

        Rules are equivalent if their representations in terms of the clusters their
        component primitives belong to are identical, or if they are length 1, and
        the _oppositve_ of subrule1's sole component falls in the same cluster as 
        subrule2's sole component.

        """
        cluster_form_1, cluster_form_2 = map(self.subrule_to_cluster_form,
                                             (subrule1, subrule2))
        if cluster_form_1 == cluster_form_2:
            return True
        elif len(subrule1) == 1 and len(subrule2) == 1:
            p = list(subrule1[0])
            if p[-2] == 'above':
                p[-2] = 'below'
            else:
                p[-2] = 'above'
            p = tuple(p)
            if p in self.primitives_to_clusters:
                if self.primitives_to_clusters[p] == cluster_form_2[0]:
                    return True
        return False

    def rule_can_join_cluster(self, cluster, subrule):
        return self.rule_pair_ok(cluster[0], subrule)

    def cluster_subrules(self):
        test = self.rule_can_join_cluster
        result = generic_cluster_from_posterior(self.posterior_by_subrule, test)
        self.subrule_clusters, self.subrule_cluster_posteriors = result
        self.subrules_to_clusters = {}
        for i,l in enumerate(self.subrule_clusters):
            for subrule in l:
                self.subrules_to_clusters[subrule] = i

    def rule_list_to_cluster_form(self, rl):
        """ Describe a rule list using the clusters its subrules belong to.
        
        Argument is a rule list expressed as a list of lists of tuples.
        
        """
        return sorted([self.subrules_to_clusters[r] for r in rl])

    def rule_list_can_join_cluster(self, cluster, rl):
        return (self.rule_list_to_cluster_form(cluster[0]) ==
                self.rule_list_to_cluster_form(rl))

    def cluster_rule_lists(self):
        test = self.rule_list_can_join_cluster
        result = generic_cluster_from_posterior(self.posterior_by_state, test)
        self.rule_list_clusters, self.rule_list_cluster_posteriors = result
        self.rule_lists_to_clusters = {}
        for i,l in enumerate(self.rule_list_clusters):
            for rule_list in l:
                self.rule_lists_to_clusters[rule_list] = i

    def do_clustering(self):
        self.cluster_primitives()
        self.cluster_subrules()
        self.cluster_rule_lists()

    def cluster_table(self, N=None):
        """ Summarize the best N clusters of rule lists. 
         
        Returns a list of strings, each reporting on one cluster.

        """
        if N is None:
            N = len(self.rule_list_clusters)
        else: 
            N = min(N, len(self.rule_list_clusters))
        reports = [self.cluster_report(i) for i in xrange(N)]
        self.reports['cluster_table'] = reports
        return reports

    def cluster_report(self, i):
        """ Summarize rule list cluster i.

        Return a string.

        """
        cluster = self.rule_list_clusters[i]
        posterior = self.rule_list_cluster_posteriors[i]
        lines = []
        header = 'Rule list cluster %d: posterior probability %.3g' % (i,posterior)
        # choose representative
        l = []
        for state_as_tuple in cluster:
            l.append((self.posterior_by_state[state_as_tuple], state_as_tuple))
        l.sort(reverse=True)
        representative_rl_as_tuple = l[0][1]
        representative_beta = self.beta_by_state[representative_rl_as_tuple]
        representative_display = self.format_rl(
            representative_rl_as_tuple, 
            representative_beta
        )
        
        lines = [header, 'Representative rule list:',
                 representative_display]
        
        prediction = logit_rules.prediction(
            logit_rules.RuleList(representative_rl_as_tuple),
            np.median(representative_beta, axis=0),
            test_data=self.model.data
        )
        representative_accuracy = self.classifier_accuracy_report(prediction)

        lines = [header, 'Representative rule list:',
                 representative_display, 'Accuracy of representative rule:',
                 representative_accuracy]
  
        return '\n'.join(lines) + '\n' 


########################################
# CLUSTERING UTILITY FUNCTIONS
# 
# These few functions don't need to be methods.

def primitive_pair_ok(p1,p2,overlap_threshold=0.5):
    """ Determine whether it is okay to cluster two primitives (given as tuples.)

    Criteria: the type and direction must be the same, and either the
    variable must be the same, and at least overlap_threshold of the shorter time
    window must lie within the longer time window.

    Thresholds are currently ignored.

    """

    (var1, window1, type1, direction1, _) = p1
    (var2, window2, type2, direction2, _) = p2

    if (type1 != type2) or (direction1 != direction2):
        return False
    if var1 != var2:
        return False
    start1, stop1 = window1
    start2, stop2 = window2

    if (start1 >= stop2) or (start2 >= stop2):
        return False

    latest_start = max(start1,start2)
    first_stop = min(stop1, stop2)
    overlap = first_stop - latest_start
    min_duration = min(stop1-start1, stop2-start2)
    fraction = overlap/float(min_duration)
    if fraction < overlap_threshold:
        return False
    else:
        return True

def primitive_can_join_cluster(cluster, primitive):
    """ Determine whether it is okay to add a primitive to a cluster.
    
    cluster - list of primitives as tuple
    primitive - candidate primitive, as tuple

    The primitive may join the cluster if it matches ALL the cluster's
    current elements, as determined by primitive_pair_ok. 

    """
    
    for element in cluster:
        if not primitive_pair_ok(element, primitive):
            return False
    return True

def generic_cluster_from_posterior(items_to_probability, test):
    """ Cluster items according, tracking associated probabilities.
    
    Arguments:
    items_to_probability: dict
    test: test(l,item) must return True if the item may be added to the
    cluster represented by list of items l, otherwise False

    Returns: 
    clusters - list of lists of items
    probabilities - list of cumulative probabilities of each cluster.

    These lists will be sorted so that the highest probability cluster is first.

    """
    l = [(v,k) for k,v in items_to_probability.iteritems()]
    l.sort()
    clusters = []
    cluster_posteriors = []
    while l:
        posterior, seed = l.pop()
        new_cluster = [seed]
        logger.debug('New cluster:')
        logger.debug(new_cluster)
        candidate_index = len(l) - 1
        while candidate_index > 0:
            candidate_posterior, candidate = l[candidate_index]
            if test(new_cluster, candidate):
                l.pop(candidate_index)
                new_cluster.append(candidate)
                posterior += candidate_posterior
                logger.debug('adds %s' % str(candidate))
            candidate_index -= 1
        clusters.append(new_cluster)
        cluster_posteriors.append(posterior)

    result = zip(cluster_posteriors, clusters)
    result.sort(reverse=True)
    cluster_posteriors, clusters = zip(*result)
    return clusters, cluster_posteriors
    

    
