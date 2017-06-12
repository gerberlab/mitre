""" Utilities for visualizing rule input data. """

import numpy as np
import matplotlib.pyplot as plt
from .rules import DiscreteRulePopulation
from .data_processing import transforms
import scipy.stats
from sklearn.metrics import normalized_mutual_info_score as normalized_mi
from .rules import PrimitiveRule as PrimitiveRule


# also import transforms, PrimitiveRule as PrimitiveRule

########################################
# RULE VALUE PANELS

class RulePaneler:
    def __init__(self, dataset, tmin, N_intervals):
        # Lazily generate rule intervals by creating a subset of the 
        # data with only one variable but the same temporal sampling patterns
        # and making DiscreteRulePopulation do it.
        self.dataset = dataset
        data_subset = transforms.select_variables(dataset,[0])
        population = DiscreteRulePopulation(None, tmin, N_intervals, data=data_subset)
        self.windows = population.windows
        
    def value_panel(self, variable_name, types=['slope','average']):
        panel = []
        variable_index = self.dataset.variable_names.index(variable_name)
        for rule_type in types:
            for window in self.windows:
                values = []
                # type and threshold aren't meaningful here
                rule = PrimitiveRule(variable_index, window, rule_type, 'above', 0)
                for subject in xrange(self.dataset.n_subjects):
                    try:
                        values.append(
                            rule.evaluate(
                                self.dataset.X[subject][variable_index,:],
                                self.dataset.T[subject]
                                )
                            )
                    except ValueError:
                        values.append(np.nan)
                panel.append(values)
        return np.vstack(panel)

    def rank_panel(self, *args, **kwargs):
        values = self.value_panel(*args, **kwargs)
        return np.vstack([scipy.stats.rankdata(v) for v in values])

    def mi_panel(self, *args, **kwargs):
        ranks  = self.rank_panel(*args, **kwargs)
        panel = []
        for rank_vector in ranks:
            scores = []
            for rank in xrange(1,len(rank_vector)):
                # Recall ranks start at 1. The highest rank is uninteresting.
                scores.append(normalized_mi(self.dataset.y, rank_vector <= rank))
            panel.append(scores)
        return np.vstack(panel)
                
########################################
# Plotting utilities
def minimal_show_rule(dataset,variable_name,window_start,window_end,average=None,slope=None):
    """
    Show how a PrimitiveRule applies to a dataset.

    Like show_rule, but with less labeling.

    """
    f, ax = plt.subplots()
    ax.axvspan(xmin=window_start,xmax=window_end,color='k',alpha=0.3)
    _alternate_show(ax, dataset, variable_name)
    ymin, ymax = ax.get_ylim()
    y_center = 0.5*(ymin+ymax)
    window_center = 0.5*(window_start+window_end)
    threshold_x = np.linspace(window_start, window_end)
    if average is not None:
        ax.plot(threshold_x,average*np.ones(len(threshold_x)),'r')
    else:
        ax.plot(threshold_x,y_center+slope*(threshold_x - window_center),'r')
    return (f, ax)

def _alternate_show(ax, dataset, variable_name):
    """ no title """
    try: 
        variable_index = dataset.variable_names.index(variable_name)
    except ValueError:
        print('Variable not in dataset.')
        return
    for i in xrange(dataset.n_subjects):
        if dataset.y[i]:
            color = 'g'
        else:
            color = 'b'
        t = dataset.T[i]
        X = dataset.X[i][variable_index,:]
        # print i
        # print t
        # print X
        ax.plot(t,X,color + '-o')

###

def show_rule(dataset,variable_name,window_start,window_end,average=None,slope=None):
    """
    Show how a PrimitiveRule applies to a dataset.

    Draws the data for the specified variable, highlighting positive subjects in green.

    Shades a window covering the specified time window, and draws a line indicating the 
    threshold value or slope.
    
    One of 'average' or 'slope' must be given.

    """
    f, ax = plt.subplots()
    ax.axvspan(xmin=window_start,xmax=window_end,color='k',alpha=0.3)
    _show(ax, dataset, variable_name)
    ymin, ymax = ax.get_ylim()
    y_center = 0.5*(ymin+ymax)
    window_center = 0.5*(window_start+window_end)
    threshold_x = np.linspace(window_start, window_end)
    if average is not None:
        ax.plot(threshold_x,average*np.ones(len(threshold_x)),'r')
    else:
        ax.plot(threshold_x,y_center+slope*(threshold_x - window_center),'r')
    ax.set_ylabel('log relative abundance')
    ax.set_xlabel('time')
    return (f, ax)
        
def show(dataset,variable_name):
    f, ax = plt.subplots()
    _show(ax, dataset, variable_name)

def _show(ax, dataset, variable_name):
    try: 
        variable_index = dataset.variable_names.index(variable_name)
    except ValueError:
        print('Variable not in dataset.')
        return
    ax.set_title('%s (index %d)' % (variable_name, variable_index))
    for i in xrange(dataset.n_subjects):
        if dataset.y[i]:
            color = 'g'
        else:
            color = 'b'
        t = dataset.T[i]
        X = dataset.X[i][variable_index,:]
        # print i
        # print t
        # print X
        ax.plot(t,X,color + '-o')

def value_plot(rp, variable_name):
    average_panel = rp.value_panel(variable_name, types=['average'])
    slope_panel = rp.value_panel(variable_name, types=['slope'])
    return _multiplot(rp.dataset, variable_name, slope_panel, average_panel)

def mi_plot(rp, variable_name):
    average_panel = rp.mi_panel(variable_name, types=['average'])
    slope_panel = rp.mi_panel(variable_name, types=['slope'])
    return _multiplot(rp.dataset, variable_name, slope_panel, average_panel,
                      left_vmin=0., right_vmin=0., left_vmax=1., right_vmax=1.)

def deviation_plot(rp, variable_name, slope_cutoff=1, average_cutoff = 2.):
    average_panel = rp.value_panel(variable_name, types=['average'])
    average_panel = (average_panel.T - np.median(average_panel, axis=1)).T
    average_panel.sort()
    average_ranges = np.max(average_panel, axis=1) - np.min(average_panel, axis=1)
    average_panel = average_panel[np.argsort(average_ranges)][::-1]

    slope_panel = rp.value_panel(variable_name, types=['slope'])
    slope_panel = (slope_panel.T - np.median(slope_panel, axis=1)).T
    slope_panel.sort()
    slope_ranges = np.max(slope_panel, axis=1) - np.min(slope_panel, axis=1)
    slope_panel = slope_panel[np.argsort(slope_ranges)][::-1]

    return _multiplot(rp.dataset, variable_name, slope_panel, average_panel,
                     left_vmin = -1.0*slope_cutoff, left_vmax = slope_cutoff,
                     right_vmin = -1.0*average_cutoff, right_vmax = average_cutoff)    

def _multiplot(dataset, variable_name, left_panel_matrix, right_panel_matrix,
               left_vmin=None, left_vmax=None, right_vmin=None, right_vmax=None):
    f = plt.figure()
    upperhalf = f.add_subplot(211)
    _show(upperhalf, dataset, variable_name)
    ll = f.add_subplot(223)
    li = ll.imshow(left_panel_matrix.T, interpolation='nearest', vmin=left_vmin, vmax=left_vmax)
    f.colorbar(li, orientation = 'horizontal')
    lr = f.add_subplot(224)
    ri = lr.imshow(right_panel_matrix.T, interpolation='nearest', vmin=right_vmin, vmax=right_vmax)
    f.colorbar(ri, orientation = 'horizontal')
    return f
