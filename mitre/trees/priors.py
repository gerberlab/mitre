""" 
Calculate prior probabilities of attaching rules to tree nodes.

"""
import numpy as np
from scipy.stats import gamma
from .utilities import subtree_lengths

def node_priors(tree, a_L, b_L_scale, normalize_lengths=True):
    """ Calculate a gamma-PDF-based prior for attachment to each node.

    If normalize_lengths is True, the subtree lengths are scaled by
    the median distance from the leaves of the tree to the root. I
    have no sound theoretical argument for this; it is convenient for
    testing purposes.

    Will modify the input tree, adding a 'prior' feature to each node.

    Assigns a probability of zero to the root node. 

    Returns a dictionary mapping node names to prior probabilities
    (which are not guaranteed to be normalized, though at present they
    are.)

    """ 
    lengths = subtree_lengths(tree)
    if normalize_lengths:
        scale = np.median([tree.get_distance(n) for n in tree.get_leaves()])
        for k in lengths:
            lengths[k] /= scale
    priors = {}
    for node, length in lengths.iteritems():
        priors[node] = gamma.pdf(length,a_L,scale=b_L_scale)
    # Zero out the root node.
    priors[tree.name] = 0.
    total_prior = sum(priors.values())
    for node, length in lengths.iteritems():
        priors[node] /= total_prior
    for node_object in tree.traverse():
        node_object.add_feature('prior',priors[node_object.name])
    return priors


