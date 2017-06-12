"""
Tools for predicting outcomes from microbiome time series data.

"""
import copy
import logging
import numpy as np
from scipy.stats import poisson
from scipy.special import betaln 
from scipy.misc import logsumexp
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

logging.basicConfig(format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Define the minimum number of observations needed in a time
# window to allow it at all...
min_observations = 1; 
# or estimate its _slope_:
min_observations_slope = 2;

class PrimitiveRule:
    def __init__(self, variable, window, type_, direction, threshold):
        """ Initialize a primitive rule from parameters. 
        
        Arguments:
        variable - which variable in multi-variable data this rule
        applies to, integer
        window - what time window this rule applies over, tuple of
        real scalars
        type_ - 'slope' or 'average'
        direction - 'above' or 'below'
        threshold - real scalar
        
        """
        self.variable = variable
        self.window = window
        self.type_ = type_
        self.direction = direction
        self.threshold = threshold
        (t0, t1) = window
        self.midpoint = (t0+t1)*0.5

    def as_tuple(self):
        """ Summarize this rule as a tuple. 

        Note that primitiveRule(*rule_a.as_tuple()) is a new rule
        equivalent to rule_a. 
        
        """
        return (self.variable, self.window, self.type_,
                self.direction, self.threshold)
    
    def __repr__(self):
        return 'PrimitiveRule(%s)' % str(self.as_tuple())

    def apply(self, X, t):
        """ Determine whether the rule fires for given data.
        
        Inputs:
        
        X - Matrix of data from one subject; n_variables x (number of 
        timepoints at which observations were made for that subject)
        t - Array of timepoints corresponding to each observation for
        this subject. 

        Returns:

        True if the rule applies to this subject, otherwise False.

        """
        relevant_data = X[self.variable, :]
        value = self.evaluate(relevant_data, t)
        if self.direction == 'above':
            return value > self.threshold
        else:
            return value <= self.threshold

    def evaluate(self, x, t):
        """ Determine the critical quantity for this rule from data.

        That is, estimate the slope or average value, as appropriate,
        during this rule's time window in the given data.

        Inputs:

        x - vector of values of the relevant variable
        t - timepoints at which those values were observed or estimated

        Returns: 

        The slope or average estimate.

        If there are too few points in the time window, 
        a ValueError is raised.

        """
        (t0, t1) = self.window
        relevant_times = (t >= t0) & (t <= t1)
        t = t[relevant_times] - self.midpoint
#        print self.as_tuple()
        x = x[relevant_times] 
        n_points = len(t)
        if n_points < min_observations:
            raise ValueError('Too few time points within rule time window.')
        if (n_points < min_observations_slope) and self.type_ == 'slope':
            raise ValueError('Too few time points to calculate slope within rule time window.')

        if self.type_ == 'slope':
            A = np.ones((n_points,2))
            A[:,0] = t
            (slope, intercept), _, _, _ = np.linalg.lstsq(A,x)
            return slope

        if self.type_ == 'average':
            # At one time we really committed to the locally linear
            # approximation to the data, did a linear regression just
            # as in the slope case, and reported the intercept as the
            # average value.
            #
            # In practice, this leads to bad results- if we have two
            # points in the window, both near the beginning and close
            # together, a modest difference between them can lead to
            # absurdly large values for the intercept in the linear
            # regression. Simply averaging them is just as defensible
            # as a matter of principle, and doesn't have this problem.
            # 
            # Of course, the slope in such a case must also be
            # suspect. (There's also the issue of time windows
            # containing only a single observation for one or more
            # subjects - there the regression can't sensibly be
            # performed at all.)  Really the issue is that we do have
            # prior beliefs about the data, its dynamics, and
            # measurement error which we are not incorporating into
            # the model (e.g., by some sort of spline fit as in
            # MDSINE.)
            #
            # At minimum we may eventually want to explore some lighter-weight
            # interpolation approach. For now, this is adequate.
            return np.mean(x)

class RuleList:
    def __init__(self, rules=[], model=None):

        """ Create RuleList object from rules. 

        Arguments:

        rules - A list of zero or more lists of primitive rules,
        specified either as PrimitiveRule objects or as tuples formatted
        as the output of PrimitiveRule.as_tuple(). Default is []. Note
        that while the overall rule list may be empty, it may not
        have empty lists as elements (ie, each rule must be the conjunction
        of one or more primitive rules.) 
        model - Optionally, a RuleModel object with which this rule
        list is associated; used for printing an informative summary.

        Raises:
        ValueError if an empty rule is specified. 

        """
        self.model = model
        self.rules = []
        for input_rule in rules:
            if len(input_rule) < 1:
                raise ValueError('Each rule must contain'
                                 ' one or more primitive rules.')
            translated_rule = []
            for primitive in input_rule:
                if isinstance(primitive,PrimitiveRule):
                    translated_rule.append(primitive)
                else:
                    translated_rule.append(PrimitiveRule(*primitive))
            self.rules.append(translated_rule)

    def as_tuple(self):
        """ Convert rule list to a tuple of tuples of tuples.
        
        Useful for comparing rule lists/subrules programmatically.

        """
        return tuple([tuple([p.as_tuple() for p in subrule]) for 
                      subrule in self.rules])

    def __repr__(self):
        return 'RuleList(%s)' % str(self.rules)

    def __str__(self):
        overall_lines = ['Rule list with %d rules' % len(self)]
        for i,rule in enumerate(self.rules):
            rule_lines = []
            for p in rule:
                line = ('(%.3f,%.3f) variable %d %s %s %.4f' %
                        (p.window[0],p.window[1],p.variable,
                         p.type_,p.direction,p.threshold))
                rule_lines.append('\t' + line)
            overall_lines.append('Rule %d:' % i)
            overall_lines = overall_lines + rule_lines
        return '\n'.join(overall_lines)

    def copy(self):
        # Deep-copying the rule list unnecessarily duplicates the 
        # primitive objects, which at this point we basically never
        # modify; but this is safest, in case at some time we decide to
        # start modifying them again.
        return RuleList(copy.deepcopy(self.rules), self.model)

    def __len__(self):
        return len(self.rules)

    def __getitem__(self,i):
        return self.rules[i]
    
    def __setitem__(self,i,item):
        self.rules[i] = item
        
    def __delitem__(self,i):
        del self.rules[i]

    def apply_rules(self, X, t):
        """ Determine which rules fire for given data.

        Arguments:
        X - n_variables * n_timepoints array of data from one
        subject.
        t - vector of observation times.

        Returns: 
        
        y - 1-D Boolean array specifying whether each rule in self.rules
        applies to this subject.

        """
        rule_truths = []
        for rule in self.rules:
            this_rule_result = True
            for primitive in rule:
                if not primitive.apply(X,t):
                    this_rule_result = False
                    break
            rule_truths.append(this_rule_result)
        return np.array(rule_truths)

class DiscreteRulePopulation:
    def __init__(self, model, tmin, N_intervals, tmax=None, data=None, 
                 max_thresholds = None): 
        """ Enumerate discretized rules from dataset in context of model.

        Arguments:

        model: RuleModel instance to which this population is
            notionally connected.  May be None if 'data' is
            specified. If 'model' is specified, 'data' is ignored and
            we set self.data = model.data.

        tmin: minimum allowable rule time window (in the units used
            for observation times in self.data.)

        N_intervals: Generate atomic time windows by dividing the
            duration of the experiment recorded in self.data into this
            many equal pieces. Longer time windows consist of two or
            more consecutive such pieces.

        tmax: maximum allowable rule time window (in self.data time
              units;) default None, in which case the maximum is set
              to np.inf.
        
        data: Dataset instance to apply rules to (ignored if 'model')
            is specified.

        max_thresholds: maximum number of threshold values allowed for
            any feature. If None, this is unlimited, and the number of
            thresholds will generally equal the number of training
            subjects minus 1.

        """
        self.model = model
        if data is None:
            self.data = model.data
        else:
            self.data = data

        self.tmin = tmin 
        if tmax is None:
            self.tmax = np.inf
        else:
            self.tmax = tmax
        self.N_intervals = N_intervals

        if max_thresholds is None:
            self.max_thresholds = np.inf
        else: 
            self.max_thresholds = max_thresholds

        self.mine_rules()

        self.applied_filters = []

    def _update_truths(self):
        """ Reevaluate truth tables for all primitives in the context of self.data.

        This is a somewhat special-purpose function, used primarily for
        cross-validation; it keeps the set of allowed time windows, but 
        regenerates everything else.

        Specifically, updates self.primitive_values, self.truth_table,
        self.flat_truth, and self.base_flat_truth; resets the filters,
        then reapplies those that were listed in self.applied_filters.

        Updates self.data._primitive_result_cache to point to 
        self.truth_table.

        """
        # Recall that flat_rules contains primitives in tuple form, and
        # truth_table is a dictionary keyed by those tuples.
        self.calculate_values() # self.primitive_values
        self.update_base_rules() # self.truth_table, self.base_flat_truth
        old_applied_filters = self.applied_filters[:]
        self.reset_filter()
        for filter_method, args in old_applied_filters:
            getattr(self,filter_method)(*args)
        self.data._primitive_result_cache = self.truth_table

    def mine_rules(self):
        """ List all relevant rules in an easily accessible format. 

        Also calculate their truth values when applied to the data
        and the values that will be used to calculate a normalized 
        prior over the primitives (specifically, the weights
        assigned to each variable in the dataset, as 
        self.flat_variable_weights, and the durations of each time
        window, as self.flat_durations.) 

        Also populates: 
        self.timepoints 
        self.windows
        self.flat_rules, 
        self.flat_weights, 
        self.flat_durations,
        self.flat_truth
        self.rule_table 
        self.truth_table 
        self.n_primitives = len(truth_table) 
        self.primitive_index 

        """
        # This process proceeds in three pieces.

        # 1. First we define the allowable set of time windows (those
        # for which enough points have been provided to estimate
        # slopes or averages- note, two separate lists.)

        # 2. Then, for each combination of variable and time window,
        # we determine and store average values and slopes as
        # appropriate. (This calls the method calculate_values.)

        # 3. Finally, we list off and store the possible rules
        # (combining variable, slope/average, direction, and
        # threshold) and their truth values for each subject.
        # (This calls the method update_base_rules.) 

        # Note that at one time we compiled the rules in effect as a
        # three-dimensional array, (windows) x (variable +
        # slope/average) x (threshold + under/over), anticipating that
        # we might want to sample by updating one aspect of a rule at
        # a time.  This has not yet become critical and it will
        # require some additional work to integrate the filtering
        # mechanic, so that attribute ('rule_table') is no longer
        # updated.

        # By the end of the call we have populated the vectors
        # self.flat_rules, self.flat_weights, self.flat_durations,
        # self.flat_truth, which give us the information needed to
        # sample efficiently over the entire primitive population.  We
        # also set up self.base_flat_rules, self.base_flat_weights
        # etc., which preserve the information on the _entire_ rule
        # population, from which various subsets can be selected with
        # the various filtering methods.

        ########################################
        # Step 1. Find all allowable time windows. 
        #
        # We divide the experimental time into N_intervals equal
        # pieces.  Each rule time window includes one or more
        # consecutive such pieces.  Windows shorter than tmin, longer
        # than tmax, or which do not include at least min_observations
        # observations for all subjects, are excluded.  The approach
        # is inefficient, but typically there are less than a few
        # hundred candidate windows anyway...
        timepoints = np.linspace(self.data.experiment_start,
                                 self.data.experiment_end,
                                 self.N_intervals+1)
        self.timepoints = timepoints
        windows = []
        windows_ok_for_slope = []
        for i,t0 in enumerate(timepoints):
            for t1 in timepoints[i+1:]:
                if t1 - t0 < self.tmin:
                    continue
                if t1 - t0 > self.tmax:
                    continue
                window_long_enough = True
                this_window_ok_for_slope = True
                for subject_timepoints in self.data.T:
                    n_timepoints_in_window = np.sum(
                        (t0 <= subject_timepoints) &
                        (subject_timepoints <= t1)
                        )
                    if n_timepoints_in_window < min_observations_slope:
                        this_window_ok_for_slope = False
                    if n_timepoints_in_window < min_observations:
                        window_long_enough = False
                        break
                if window_long_enough:
                    windows.append((t0,t1))
                    windows_ok_for_slope.append(this_window_ok_for_slope)
        n_windows = len(windows)
        self.windows = windows
        self.windows_ok_for_slope = windows_ok_for_slope
        assert n_windows == len(windows_ok_for_slope)


        ########################################
        # 2. Tabulate average values and slopes in relevant regions.
        self.calculate_values()

        ########################################
        # 3. List off the rules and their properties, and determine their truth values
        self.update_base_rules()

        ########################################
        # 4. Sort rules and initialize the unfiltered arrays

        self.sort_base_rules()
        #self.reset_filter()


    def sort_base_rules(self):
        """ Sort the population lexicographically by truth vector.

        This should help speed up likelihood calculations.

        Note, resets the filter.

        """ 
        # np.lexsort will sort columns by rows, with the last
        # row as the primary sort key, etc; so we rotate the 
        # truth array by 90 degrees to get it to do what we want.
        new_order = np.lexsort(np.rot90(self.base_flat_truth))
        self._reordering_cache = new_order

        self.base_flat_durations = self.base_flat_durations[new_order]
        self.base_flat_variable_weights = self.base_flat_variable_weights[new_order]
        new_flat_rules = [self.base_flat_rules[i] for i in new_order]
        self.base_flat_rules = new_flat_rules
        self.base_flat_truth = self.base_flat_truth[new_order]
        self.base_primitive_index = {
            t:i for i,t in enumerate(new_flat_rules)
        }

        self.reset_filter()


    def update_base_rules(self):
        self.base_flat_rules = []
        self.base_flat_variable_weights = []
        self.base_flat_durations = [] 
        self.base_flat_truth = []
        flat_index = 0 
        truth_table = {}
        self.base_primitive_index = {}

        for (window, do_slope) in zip(self.windows,
                                      self.windows_ok_for_slope):
            duration = window[1] - window[0]
            for variable in xrange(self.data.n_variables):
                variable_weight = self.data.variable_weights[variable]
                for type_ in ('slope','average'):
                    if type_ == 'slope' and not do_slope:
                        continue
                    values = self.primitive_values[(window, variable, type_)]
                    # Now enumerate all the possible thresholds and
                    # iterate over them. (This function can return
                    # data associated with each threshold if we choose
                    # to add any. Thresholds will be unique. In some
                    # instances the list of possible thresholds may be
                    # empty.)
                    thresholds = self.thresholds_from_values(values)
                    for t in thresholds:
                        for direction in ('above','below'):
                            primitive = (variable, window, type_,
                                         direction, t)
                            if direction == 'above':
                                truth = values > t
                            else:
                                truth = values <= t
                            truth_table[primitive] = truth
                            # Now handle adding to all the flat 
                            # collections
                            self.base_flat_rules.append(primitive)
                            self.base_flat_variable_weights.append(variable_weight)
                            self.base_flat_durations.append(duration)
                            self.base_flat_truth.append(truth_table[primitive])
                            self.base_primitive_index[primitive] = flat_index
                            flat_index += 1 
        self.truth_table = truth_table
        self.base_flat_variable_weights = np.array(self.base_flat_variable_weights)
        self.base_flat_durations = np.array(self.base_flat_durations)
        self.base_flat_truth = np.vstack(self.base_flat_truth) 
        # should be dtype=bool without any intervention...

    def thresholds_from_values(self, values, delta=None):
        """ Given sampled values for a feature, generate threshold values.

        If the number of values is less than or equal to
        self.max_thresholds + 1, and no delta is given, or all the
        points are spaced more widely than delta, the thresholds are
        the midpoints between each pair of values.

        If there are more values than self.max_thresholds + 1, or
        delta is given and at least one pair of values differ by less
        than delta, we use an agglomerative clustering approach with
        the UPGMA metric. If we reach self.max_thresholds + 1 clusters
        and find the distance between all clusters is at least delta,
        clustering stops; otherwise, clustering continues until that
        condition is met. The thresholds are then the midpoints
        between the upper edge of each cluster and the lower edge of
        the cluster above. If there is only one cluster, no thresholds
        are returned. (Note that in some cases this may produce
        _thresholds_ which differ by less than delta.)

        In either case, only unique values are returned. 

        """
        values = np.sort(values)
        do_clustering = True
        if (len(values) <= self.max_thresholds + 1):
            thresholds = 0.5*(values[:-1] + values[1:])
            if not delta:
                do_clustering = False
            else:
                differences = values[1:]-values[:-1]
                if np.min(differences) > delta:
                    do_clustering = False

        if do_clustering:
            distances = []
            for i, v1 in enumerate(values):
                for v2 in values[i+1:]:
                    distances.append(v2-v1)
            distances = np.array(distances)
            linkage = hierarchy.linkage(distances, method='average')
            n_clusters = min(self.max_thresholds + 1, len(values))
            clusters = hierarchy.fcluster(linkage,
                                          n_clusters,
                                          criterion='maxclust')
            if delta:
                # Generate an alternative clustering with minimum
                # distance delta between clusters. One clustering
                # option will be nested in the other. Use the coarser
                # clustering of the two.
                clusters_dist = hierarchy.fcluster(linkage, 
                                                   delta,
                                                   criterion='distance')
                # n_clusters_max should always equal n_clusters...
                n_clusters_max = np.max(clusters)
                n_clusters_dist = np.max(clusters_dist)
                if n_clusters_dist < n_clusters_max:
                    clusters = clusters_dist
            boundaries = (clusters[:-1] != clusters[1:])
            upper_edges = values[:-1][boundaries]
            lower_edges = values[1:][boundaries]
            thresholds = 0.5*(upper_edges + lower_edges)
            
        return np.unique(thresholds)

    def calculate_values(self):
        """ Tabulate average values and slopes in relevant regions. """
        # primitive_values[(window, variable, slope/average)] = vector of floats per subject
        
        primitive_values = {}
        
        subject_info = zip(self.data.X, self.data.T)
        for (window, do_slope) in (zip(self.windows,
                                       self.windows_ok_for_slope)):
            for variable in xrange(self.data.n_variables):
                for type_ in ('slope','average'):
                    if type_ == 'slope' and not do_slope:
                        continue
                    base_rule = PrimitiveRule(variable, window, type_,
                                              'above', 0.)
                    this_combination_values = []
                    for data, timepoints in subject_info:
                        this_combination_values.append(base_rule.evaluate(data[variable],
                                                                          timepoints))
                    primitive_values[(window, variable, type_)] = (
                        np.array(this_combination_values)
                        )
        self.primitive_values = primitive_values

    def reset_filter(self):
        """ Load all possible primitives back into the active population again. """
        
        self.flat_rules = self.base_flat_rules[:] # list
        self.flat_variable_weights = self.base_flat_variable_weights # 1d array
        self.flat_durations = self.base_flat_durations # 1d array 
        self.flat_truth = self.base_flat_truth # 2d array
        self.n_primitives = len(self.flat_rules)
        self.primitive_index = self.base_primitive_index
        self.applied_filters = []

    def _filter(self, boolean_index):
        """ Keep rules active or exclude them based on entries in a list/vector. 

        Filtering is applied to the current active rules; thus boolean_index
        should be as long as self.flat_rules at the time this method is called.

        """
        self.flat_rules = [primitive for primitive, include in 
                           zip(self.flat_rules, boolean_index) if include]
        self.flat_variable_weights = self.flat_variable_weights[boolean_index]
        self.flat_durations = self.flat_durations[boolean_index]
        self.flat_truth = self.flat_truth[boolean_index, :]
        self.n_primitives = len(self.flat_rules)
        index = {}
        for i, primitive in enumerate(self.flat_rules):
            index[primitive] = i
        self.primitive_index = index

    def filter_thresholds(self, min_difference_average, min_difference_slope,
                          min_difference_exp_average=None, min_difference_exp_slope=None):
        """ Exclude thresholds dividing groups of subjects which don't differ enough.

        Specifically, for every combination of variable, time window,
        and type (slope/average), divide the values into groups
        separated by the thresholds currently present in the pool. Do
        an agglomerative clustering of those bins based on their
        median values. Walk down the tree from the root, allowing the
        threshold which splits each node's children only if the the
        medians of the bins associated with each child cluster differ
        by more than min_difference_average or min_difference_slope,
        as appropriate, in a UPGMA sense (or if the medians of the
        exponentials of the medians of the bins differ by more than
        min_difference_exp_average, etc.)

        """
        self.applied_filters.append(
            ('filter_thresholds',
             (min_difference_average, min_difference_slope,
              min_difference_exp_average, min_difference_exp_slope))
        )
        keep = np.ones(len(self.flat_rules), dtype=np.bool_)
        thresholds_by_key = {}
        indices = {} 
        for i, (variable, window, type_, _, threshold) in enumerate(self.flat_rules):
            key = (window, variable, type_) 
            thresholds_by_key.setdefault(key, set()).add(threshold)
            indices.setdefault(key, set()).add(i)
        for key, thresholds in thresholds_by_key.iteritems():
            logger.debug('Filtering thresholds for key %s ' % str(key))
            type_ = key[-1]
            if type_ == 'average':
                difference_cutoff = min_difference_average
                exp_difference_cutoff = min_difference_exp_average
            if type_ == 'slope':
                difference_cutoff = min_difference_slope
                exp_difference_cutoff = min_difference_exp_slope
            values = self.primitive_values[key]
            values = np.sort(values) 
            edges = [-np.inf] + sorted(list(thresholds)) + [np.inf]
            bin_medians = []
            logger.debug('Values/edges:')
            for v in [values, edges]:
                logger.debug(str(v))
            for bin_start, bin_end in zip(edges[:-1],edges[1:]):
                # In edge cases where several values are equal (due to
                # eg multiple observations with the same number of
                # counts for an OTU) the behavior may become somewhat
                # pathological and a bin may be empty, typically
                # because all the values that would be assigned to it
                # are piled up on its upper edge. In that case we let
                # its median be the upper edge (though in the longer
                # term we might want to use eg the bin midpoint,
                # rather than the median of the data within the bin,
                # for all the bins... but that will require some
                # attention to the cases where one edge is infinite.)
                # We may end up with threshold values that lead to
                # rules with identical truth values, but this should
                # not happen often enough to be problematic. 
                bin_indices = np.logical_and(bin_start <= values, values < bin_end)
                logger.debug('Bin start/end/indices:')
                logger.debug('%.3g' % bin_start)
                logger.debug('%.3g' % bin_end)
                logger.debug(str(bin_indices))
                if sum(bin_indices) == 0:
                    logger.debug('Empty bin, using upper edge as median '
                                 'and continuing')
                    bin_medians.append(bin_end)
                    continue
                bin_medians.append(np.median(values[bin_indices]))
            bin_medians = np.array(bin_medians)
            logger.debug('Bin medians:')
            logger.debug(str(bin_medians))
            distances = []
            for i, median1 in enumerate(bin_medians):
                for median2 in bin_medians[i+1:]:
                    distances.append(median2-median1)
            distances = np.array(distances)
            # This should be the needed 'condensed form of the distance matrix'.
            # Now we form a hierarchical clustering...
            linkage = hierarchy.linkage(distances, method='average')
            # Walk the resulting tree upward from the nodes immediately above
            # the leaves towards the root. At each step: 
            # 
            # 1. record which leaves are associated with the cluster defined by the node
            # 2. calculate the distance between the medians of the child clusters, on
            # a normal and, if requested, exponential scale
            # 3. if that distance exceeds the appropriate cutoff, determine which threshold
            # separates the clusters, and add it to a list of `good' thresholds.
            good_thresholds = set()
            # For convenience, prepopulate the mapping with single-leaf 'clusters'
            # that are the leaves themselves.
            node_to_leaves = {i: [i] for i in xrange(len(bin_medians))}

#            print 'remove this debug output'
#            print linkage 
            for row_index, (child1, child2, _, _) in enumerate(linkage):
                node = row_index + len(bin_medians)
                logger.debug('%s: inner node %d' % (key, node))
                split_ok = False
                left_leaves = node_to_leaves[int(child1)]
                right_leaves = node_to_leaves[int(child2)]
                # The assignment to 'left' and 'right' above is
                # arbitrary- give it meaning by swapping them if
                # needed to ensure that the left leaves are smaller
                # than the right.
                if left_leaves[-1] > right_leaves[0]:
                    temp = left_leaves
                    left_leaves = right_leaves
                    right_leaves = temp

                logger.debug('left leaves: %s' % str(left_leaves))
                logger.debug('right leaves: %s' % str(right_leaves))

                node_to_leaves[node] = left_leaves + right_leaves
                difference = (np.median(bin_medians[right_leaves]) - 
                              np.median(bin_medians[left_leaves]))
                logger.debug('difference: %.3g' % difference)
                if difference >= difference_cutoff:
                    split_ok = True
                if exp_difference_cutoff is not None:
                    exp_difference = (
                        np.median(np.exp(bin_medians[right_leaves])) - 
                        np.median(np.exp(bin_medians[left_leaves]))
                    )
                    logger.debug('exp_difference: %.3g' % exp_difference)
                    if exp_difference >= exp_difference_cutoff:
                        split_ok = True
                if not split_ok:
                    logger.debug('Difference not large enough, continuing')
                    continue
                # left_leaves[-1] is the index, say J, of the highest bin in the left
                # child cluster; that bin's upper edge is defined by edges[J+1].
                # Note this will never be edges[-1].
                good_thresholds.add(edges[left_leaves[-1]+1])
                logger.debug('Added threshold %d: %.3g' % (left_leaves[-1],
                                                           edges[left_leaves[-1]+1]))
            
            # Finally, look up all the primitives corresponding to
            # this key, and mark them for removal if their threshold
            # is not approved. If multiple primitives have the same
            # threshold, ensure only one is kept. (This should not
            # happen, because we will not generally generate
            # non-unique rules; thus this is redundant and can be
            # removed.)
            logger.debug('Final set of allowed thresholds: %s' % str(sorted(good_thresholds)))
            already_used_threshold_direction_pairs = set()
            for primitive_index in indices[key]:
                threshold = self.flat_rules[primitive_index][-1]
                direction = self.flat_rules[primitive_index][-2]
                if threshold not in good_thresholds:
                    logger.debug('Discarding %s' % str(self.flat_rules[primitive_index]))
                    keep[primitive_index] = False
                elif (threshold, direction) in already_used_threshold_direction_pairs:
                    logger.debug('Discarding duplicate %s' % str(self.flat_rules[primitive_index]))
                    keep[primitive_index] = False
                else:
                    already_used_threshold_direction_pairs.add((threshold,direction))

        self._filter(keep)
        
    def get_primitive_index(self, primitive):
        return self.primitive_index.get(primitive.as_tuple())

    def sort_list_of_primitives(self,l):
        """ Sort list of PrimitiveRule objects by ordering on this population.

        Sorting is done in place.

        PrimitiveRules which do not belong to the population will lead
        to an exception.

        No return value.

        """
        l.sort(key=self.get_primitive_index)

    def __len__(self):
        return self.n_primitives

###

class RuleModel(object):
    """ Base class for rule list/collection models.

    """
    def __init__(self,
                 data,
                 lambda_rules, 
                 lambda_primitives,
                 tmin,
                 **kwargs):
        """ Parent class for rule list/collection models.

        Arguments:
        data - a Dataset object; in this formulation, the model
        is conditional on the observed data.
        lambda_rules - parameter for Poission prior on total rule
        list length
        lambda_primitives - parameter for Poisson prior on the length
        (i.e., number of primitives) of each individual rule
        tmin - minimum allowable rule time window. Note in practice
        the minimum time window may be longer than this, to ensure
        the window of any valid rule encompasses at least a minimum
        number of observation points in each subject.

        """
        
        self.data = data
        self.lambda_rules = lambda_rules
        self.lambda_primitives = lambda_primitives
        self.tmin = tmin

    def prior(self, rule_list):
        """ Evalute the prior log-probability of a rule list.

        Should be properly normalized.

        """
        raise NotImplementedError
        # primitive_piece = sum([self.primitive_prior(primitive) for
        #                        rule in rule_list.rules for primitive in
        #                        rule])
        # structure_piece = self.structure_prior(rule_list)
        # return primitive_piece + structure_piece 
    
    def structure_prior(self, rule_list, empty_probability, lambda_rules,
                        lambda_primitives):
        """ Evaluate the log prior probability of a rule list structure.

        Assumes Poisson priors on the overall and subrule lengths
        (strictly those lengths minus 1) if the list is not empty, and
        an overall prior probability of the list being empty.

        At the moment, maintains a slightly awkward agnosticism as to
        whether the Poisson parameters are fixed attributes of the
        model, or variables, by accepting them as kwargs but looking
        them up as attributes if not provided.

        Properly normalized.

        """
        # if lambda_rules is None:
        #     lambda_rules = self.lambda_rules
        #     raise ValueError('debug')
        # if lambda_primitives is None:
        #     lambda_primitives = self.lambda_primitives

        rules = rule_list.rules
        if len(rules) == 0:
            return np.log(empty_probability)
        else:
             l0 = np.log(1-empty_probability)      
             l1 = poisson.logpmf(len(rules)-1,lambda_rules);
             # Though the overall structure prior is not normalized, we do
             # need to be careful here about the normalization of the prior
             # on the number of primitives in each rule (as a constant
             # factor introduced there will be repeated a variable number
             # of times for rules of different lengths, making the
             # structure priors not comparable.
             l2 = sum([poisson.logpmf(len(rule)-1, lambda_primitives) for 
                       rule in rules])
        return l0 + l1 + l2

    def prior_contribution_from_primitives(self, rule_list):
        """ Get contribution to log prior of this rl from choice of primitives.

        """
        # Illustrative example implementation: typically not 
        # the optimal way to calculate this value in practice.
        return sum([self.primitive_prior(p) for rule in rule_list for p 
                    in rule])

    def primitive_prior(self, primitive):
        """ Get log prior probability of filling a slot with this primitive.
        
        Exactly how this should behave is not well-defined yet. 

        """
        raise NotImplementedError()

    def primitive_variable_prior(self, variable):
        raise NotImplementedError()

    def primitive_window_prior(self, window):
        """ Exactly how this should behave is not well-defined yet. """
        raise NotImplementedError()

    def primitive_threshold_prior(self, threshold):
        """ Exactly how this should behave is not well-defined yet. """
        raise NotImplementedError()

class DiscreteRuleModel(RuleModel):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()
        
    def flat_prior(self):
        """ Evaluate log-probability of each primitive in the population.

        Return value is properly normalized.
        
        In this base class, we implement a version where the 
        distribution over primitives is static; subclasses will 
        reevaluate this at each call based on the values of variables and 
        parameters.

        """

        raw_weights = np.zeros(len(self.rule_population))
        normalization = logsumexp(raw_weights)
        return raw_weights - normalization
###

class Dataset:
    def __init__(self, X, T, y, 
                 variable_names, variable_weights,
                 experiment_start, experiment_end,
                 subject_IDs=None, subject_data=None,
                 additional_subject_categorical_covariates=[],
                 additional_covariate_default_states=[],
                 additional_subject_continuous_covariates=[],
                 variable_annotations={}):
        """ Store experimental data in an object.

        We assume that most preprocessing (variable selection, creation
        of aggregate data for higher taxa, rescaling and renormalization,
        etc.) has been done already, and that some prior on the 
        likelihood of rules in the model applying to each variable
        has already been calculated. 

        Arguments:

        X - list, containing for each subject an n_variables by 
        n_timepoints_for_this_subject array of observations.
        T - list, containing for each subject a vector giving the
        observation timepoints (note the length will be different for
        each subject in general.)
        y - list/array of boolean or 0/1 values indicating whether each 
        subject developed the condition of interest. 
        variable_names - list of strings, used for formatting output
        variable_weights - array of weights for each variable which will
        be used later to calculate the prior probability that a rule applies
        to that variable. 
        experiment_start - Time at which the experiment started.
        experiment_end - Time at which the experiment ended. 
        subject_IDs - list of identifiers for each experimental subject
        (currently purely for reference)
        subject_data - Optional pandas dataframe giving additional information 
        about each subject. 
        additional_subject_categorical_covariates - list, optional (see below)

        additional_covariate_default_states - list, optional. If these
        two arguments are given, additional covariates are included in
        the logistic regression, which do not depend on the rule
        set. The covariates should be the names of columns in the
        dataframe passed as subject_data. For each column, a list of
        unique values is generated.  For each unique value OTHER than
        the corresponding entry in additional_covariate_default_states,
        a new covariate is generated, which is 1 for subjects for whom
        the feature takes that value, and 0 otherwise (effectively, a
        one-hot encoding leaving out the default value.) The matrix of 
        covariates is stored in self.additional_covariate_matrix, and a
        list of (feature, value) pairs corresponding to each column is 
        stored in self.additional_covariate_encoding.         

        additional_subject_continuous_covariates - list, optional. If
        this argument is given, additional covariates are included in
        the logistic regression, which do not depend on the rule set.
        The covariates should be the names of columns in the dataframe
        passed as subject_data. For each column, a list of unique
        values is generated. Unlike the categorical covariates, no
        default state is specified and no encoding is performed-
        instead, each specified column in the dataframe is adjusted to
        have zero mean and then inserted directly into the regression
        covariate matrix. (Note that no rescaling is done, and the
        mean-centering behavior is only a convenience- fundamentally,
        it is the user's responsibility to transform the data as they
        feel is appropriate before incorporating it into the
        regression! In particular, the user should consider the scale
        of this covariate relative to the columns in the covariate
        matrix which come from the rule set or the discrete
        covariates- all entries in those columns will be either 0 or
        1.) Note that these values _should never be NaN_. This is not
        explicitly checked for at this stage for consistency with the
        behavior of the main.py script for categorical variables:
        subjects with NaNs in relevant columns are dropped _after_
        initial generation of the dataset and then the covariate
        matrix is regenerated. A later release may revisit this
        decision.

        variable_annotations: dict mapping variable names to taxonomic or 
        other descriptions, optional

        This method sets up the following useful attributes:
        self.n_subjects
        self.n_variables
        self.n_fixed_covariates (equal to 1 (for the constant term) + the number
        of columns in self.additional_covariate_matrix)

        Dataset objects also offer convenience methods
        apply_primitive, apply_rules, and stratify, which determine
        the output of primitives, rules, or rule lists applied to the
        data. This allows for various caching approaches that speed up
        these evaluations: currently implemented as an attribute
        _primitive_result_cache, a dict (empty by default) of arrays
        of booleans giving the truths of primitives, expressed as
        tuples, for each subject in the dataset.

        Raises ValueError if the number of variables reported
        on in each observation table in X differs from the number
        of variable names provided, or if that number does not
        match the dimension of the argument variable_weights.

        """
        self.X = X 
        self.T = T 
        self.y = np.array(y,dtype='bool')
        self.variable_names = variable_names
        self.variable_weights = variable_weights
        self.experiment_start = experiment_start
        self.experiment_end = experiment_end
        self.subject_data = subject_data
        self.subject_IDs = subject_IDs
        self.n_subjects = len(X)
        self.n_variables = len(variable_weights)
        self._primitive_result_cache = {}
        for array in X:
            # Allow a special case where a subject has no
            # observations
            if len(array) == 0:
                continue
            this_subject_n_variables, _ = array.shape
            if this_subject_n_variables != self.n_variables:
                raise ValueError('Observation-prior dimension mismatch.')
        if len(self.variable_names) != self.n_variables:
            raise ValueError('Incorrect number of variable names.')

        self.additional_subject_categorical_covariates = additional_subject_categorical_covariates
        self.additional_covariate_default_states = additional_covariate_default_states
        self.additional_subject_continuous_covariates = additional_subject_continuous_covariates
        self.generate_additional_covariate_matrix()
        self.variable_annotations = variable_annotations

    def generate_additional_covariate_matrix(self):
        """ Encode additional covariate matrix at initialization.

        See __init__.

        """
        
        columns = []
        explanations = []
        
        # First, handle the categorical covariates.
        features_and_defaults = zip(
            self.additional_subject_categorical_covariates,
            self.additional_covariate_default_states
        )

        for feature, default_value in features_and_defaults:
            try:
                values = set(self.subject_data[feature])
            except KeyError():
                # The default exception message benefits from
                # a little context here
                raise KeyError('Trying to control for covariate %s, but no "%s" '
                               'column in subject_data.' % (feature, feature))
            try:
                other_values = values.copy()
                other_values.remove(default_value)
            except KeyError:
                raise ValueError('Trying to control for covariate %s, but no '
                                 'subject has the default value "%s". To avoid '
                                 'identifiability problems, at least one subject '
                                 'must have the default value.' %
                                 (feature, default_value))
            # With that out of the way...
            for alternative in other_values:
                explanations.append((feature, alternative))
                columns.append(
                    (self.subject_data[feature] == alternative).values.astype('int64')
                )
        # Second, the continuous covariates.
        for feature in self.additional_subject_continuous_covariates:
            try:
                values = self.subject_data[feature].values
            except KeyError():
                # The default exception message benefits from
                # a little context here
                raise KeyError('Trying to control for covariate %s, but no "%s" '
                               'column in subject_data.' % (feature, feature))
            columns.append(values - np.mean(values))
            explanations.append('continuous feature %s' % feature)

        self.additional_covariate_encoding = tuple(explanations)
        if columns:
            self.additional_covariate_matrix = np.vstack(columns).T
        else:
            self.additional_covariate_matrix = None
        self.n_fixed_covariates = 1 + len(columns)

    def copy(self):
        return copy.deepcopy(self)

    def apply_rules(self, rule_list):
        """ Tabulate which rules apply to which subjects.

        Returns an n_rules x n_subjects array of booleans.

        """
        # Don't use ufunc.reduce here, it is slower
        rule_results = [
            reduce(np.logical_and, [self.apply_primitive(p) for p in rule]) for
            rule in rule_list.rules
        ]
        if rule_results:
            return np.vstack(rule_results)
        else:
            return []

    def covariate_matrix(self, rule_list):
        """ Calculate covariate matrix resulting when a rule list is applied to this data.

        A convenience function, returning
        self.apply_rules(rule_list).T after casting it to integer
        dtype (from boolean)-- this is important for cases where the
        dot product of the covariate matrix with its transpose is
        taken, for example-- and appending a column of 1s
        (representing the 'default rule' or a constant term in the
        logistic regression case) as well as self.additional_covariate_matrix
        (if applicable).

        If the rule list is empty, the matrix includes only the constant terms.

        """
        if len(rule_list) < 1:
            X = np.ones((self.n_subjects, 1),dtype=np.int64)

        else:
            # Multiplying by 1 promotes the matrix to integer. The hstack
            # might too, though I have not checked this
            X = np.hstack((1 * self.apply_rules(rule_list).T, 
                           np.ones((self.n_subjects, 1),dtype=np.int64),)
                          )
        if self.additional_covariate_matrix is not None:
            X = np.hstack((X,self.additional_covariate_matrix))
            
        return X

    def apply_primitive(self, primitive):
        if primitive.as_tuple() in self._primitive_result_cache:
            return self._primitive_result_cache[primitive.as_tuple()]
        else:
            return self._apply_primitive(primitive)

    def _apply_primitive(self, primitive):
        values = []
        for subject_x,subject_t in zip(self.X, self.T):
            values.append(primitive.apply(subject_x,subject_t))
        return np.array(values)                          

    def stratify(self, rule_list):
        subjects_handled = np.zeros(self.n_subjects, 
                                    dtype='bool')
        class_memberships = np.zeros((len(rule_list.rules) + 1,
                                      self.n_subjects),
                                     dtype='bool')
        i = -1 # need to make this explicit for the empty-rule-list case
        for i, rule_results in enumerate(self.apply_rules(rule_list)):
            this_rule_subjects = rule_results & (~subjects_handled)
            class_memberships[i,:] = this_rule_subjects
            subjects_handled = subjects_handled | this_rule_subjects
        class_memberships[i+1,:] = ~subjects_handled
        return class_memberships

    def y_by_class(self,rule_list):
        class_memberships = self.stratify(rule_list)
        return zip(np.sum(class_memberships*self.y,1),
                   np.sum(class_memberships,1))

    def __str__(self):
        template = ('<Dataset with %d subjects ' +
                    '(%d observations of %d variables)>')
        n_obs = sum([len(timepoints) for timepoints in self.T])
        return template % (self.n_subjects, n_obs,
                           self.n_variables)
    
def generate_y_logistic(data, rule_list, coefficients, offset=0.):
    """ Simulate random outcomes y based on a logistic model.

    Multiplies data.covariate_matrix(rule_list) by 
    coefficients to obtain linear predictors eta, adds offset,
    transforms to obtain probabilities, draws outcomes y with those
    probabilities, updates data.y, returns the probabilities. 

    """
    X = data.covariate_matrix(rule_list)
    eta = np.dot(X,coefficients)
    eta = eta + offset
    probabilities = np.exp(eta)/(1.0+np.exp(eta))
    outcomes = np.random.rand(*eta.shape) < probabilities
    data.y = outcomes 
    return (eta, outcomes, probabilities)
    
        
### 

class RuleListSampler:
    def sample(self, N):
        for i in xrange(N):
            logger.info('Sampling iteration %d' % i) 
            self.step();

def structure_string(rl):
    # We may want to go with the original BRL 
    # approach here and consider only the number of primitives, or 
    # number of rules then number of primitives. For the moment, stay consistent with 
    # the MATLAB version:
    # Ignore time windows and threshold values, consider all fields, sort primitives 
    # within each rule.
    rule_strings = []
    for rule in rl.rules:
        primitive_strings = []
        for primitive in rule:
            primitive_strings.append(str((primitive.variable,primitive.type_,primitive.direction)) + '\n')
            primitive_strings.sort()
        rule_strings.append(''.join(primitive_strings))
    return '--\n'.join(rule_strings)

def choose_from_relative_probabilities(p):
    p = np.cumsum(p)
    marker = np.random.rand()*p[-1]
    return np.sum(p<marker)

def match_features(base_rl, target_rl, variable_prior_array, 
                   variable_similarity_matrix):
    """ Find, quantify best match in target for each primitive in base.

    Returns an array. First column is the index of the base rule in 
    which a primitive is present, second the prior probability
    of attachment to the relevant variable, third the best similarity
    score. If multiple targets are specified, the third column gives
    scores from the first target, and subsequent columns etc.

    One row is returned for each primitive in base_rl.

    Note that nothing prevents multiple primitives in the base
    list from being matched to one primitive in a target list.

    See primitive_similarity_score for an explanation of the score.

    Arguments:
    base_rl: rule list whose primitives are sought in the target (e.g.
        a ground truth rule list)
    target_rl: rule list to search (e.g. a rule list recovered by sampling),
        or list of rule lists
    variable_prior_array: priors on each variable 
    variable_similarity_matrix: entry i,j gives a similarity (0-1) 
        for variables i and j. 

    """
    if not isinstance(target_rl, list):
        target_rl = [target_rl]
    rows = []
    for i,rule in enumerate(base_rl.rules):
        for p in rule:
            row = [i,variable_prior_array[p.variable]]
            for t in target_rl:
                scores = []
                for target_rule in t:
                    for target_p in target_rule:
                        score = primitive_similarity_score(
                            p, target_p,
                            variable_similarity_matrix
                        )
                        scores.append(score)
                scores.append(0) # handle the empty-rl case
                row.append(max(scores))
            rows.append(row)
#    return rows
    return np.vstack(rows)

def match_pairs(base_rl, target_rl, variable_prior_array, 
                   variable_similarity_matrix):
    """ Quantify best match in target for each pairwise interaction in base.

    Returns an array. First column is the index of the base rule in
    which a pair of primitives is present, second and third the prior
    probabilities of attachment to the relevant variable, fourth the best
    similarity score. If multiple targets are specified, the fourth
    column gives scores from the first target, and subsequent columns
    etc.

    One row is returned for each unique pair of primitives in each
    rule in base_rl. If there are no pairwise interactions in base_rl,
    returns [].

    Note that nothing prevents multiple pairs of primitives in the base
    list from being matched to one pair of primitives in a target list;
    however, the primitives in a base pair must be matched to two 
    different primitives, in the same rule, in the target list.

    The similarity score is then the product of the similarities
    of the base primitives to their images in the target.

    See primitive_similarity_score for an explanation of the score.

    Arguments:
    base_rl: rule list whose primitives are sought in the target (e.g.
        a ground truth rule list)
    target_rl: rule list to search (e.g. a rule list recovered by sampling),
        or list of rule lists
    variable_prior_array: priors on each variable 
    variable_similarity_matrix: entry i,j gives a similarity (0-1) 
        for variables i and j. 

    """
    if not isinstance(target_rl, list):
        target_rl = [target_rl]
    rows = []
    for i,rule in enumerate(base_rl.rules):
        for j,p1 in enumerate(rule):
            for p2 in rule[j+1:]:
                row = [i,variable_prior_array[p1.variable],
                         variable_prior_array[p2.variable]]
                for t in target_rl:
                    scores = [0,] # add a zero, handling case where t is empty
                    for target_rule in t:
                        if len(target_rule) < 2:
                            continue
                        s1 = []
                        s2 = []
                        for target_p in target_rule:
                            s1.append(
                                primitive_similarity_score(
                                    p1, target_p, 
                                    variable_similarity_matrix
                                )
                            )
                            s2.append(
                                primitive_similarity_score(
                                    p2, target_p, 
                                    variable_similarity_matrix
                                )
                            )
                        i1 = np.argmax(s1)
                        i2 = np.argmax(s2)
                        if i1 != i2:
                            scores.append(s1[i1] * s2[i2])
                        else:
                            score1 = s1[i1] * max(s2[:i1] + s2[i1+1:])
                            score2 = s2[i2] * max(s1[:i2] + s1[i2+1:])
                            scores.append(max(score1,score2))
                        scores.append(0) # in case there are no valid candidates
                    row.append(max(scores))
                rows.append(row)

    if not rows:
        return []
    return np.vstack(rows)

def primitive_similarity_score(p1, p2, variable_similarity_matrix):
    if p1.type_ != p2.type_:
        return 0
    variable_similarity = variable_similarity_matrix[p1.variable, p2.variable]
    p1min, p1max = p1.window
    p2min, p2max = p2.window
    p1duration = p1max - p1min
    p2duration = p2max - p2min
    if p1max < p2min or p2max < p1min:
        window_similarity = 0
    else:
        if p1min <= p2min and p2max <= p1max:
            overlap = p2duration
        elif p2min <= p1min and p1max <= p2max:
            overlap = p1duration 
        elif p1min <= p2min and p1max <= p2max:
            overlap = (p1max - p2min)
        elif p2min <= p1min and p2max <= p1max:
            overlap = (p2max - p1min)
        else:
            raise
        window_similarity = overlap/max(p1duration, p2duration)
    return window_similarity * variable_similarity
    
    
