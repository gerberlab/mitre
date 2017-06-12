""" Utilities for working with phylogenetic trees. 

Includes code for styling trees for ete3's drawing capabilities.

"""

import numpy as np
import ete3
from ete3 import Tree
from ..rules import Dataset

def subtree_lengths(tree):
    """ 
    Calculate total lengths of subtrees descending from each node tree.

    Assumes no duplicate node names.

    """

    lengths = {}
    # The levelorder strategy sorts from the root outwards
    # to the farthest leaves; reversing this should guarantee
    # we reach a node's children before it
    nodes = [n for n in tree.traverse(strategy='levelorder')] 
    nodes.reverse()
    for n in nodes:
        d = sum([lengths[c.name] for c in n.get_children()])
        if n.up:
            d += n.dist
        lengths[n.name] = d
    return lengths

def node_similarities(tree, node_ordering):
    n_nodes = len(tree.get_descendants()) + 1
    S = np.zeros((n_nodes, n_nodes))
    native_ordering = []
    # Start from the leaves and work backwards, then fill 
    # in the other half of the matrix
    nodes = [n for n in tree.traverse(strategy='levelorder')] 
    nodes.reverse()
    for i,node in enumerate(nodes):
        native_ordering.append(node.name)
        S[i,i] = 1
        for c in node.get_children():
            j = native_ordering.index(c.name)
            S[i,:] += 0.5 * S[j,:]
    S = S + S.T
    # now we have over-counted the diagonals...
    for i in xrange(len(nodes)):
        S[i,i] = 1
    # Reorder.
    S_reordered = np.zeros((len(node_ordering), len(node_ordering)))
    for i,name1 in enumerate(node_ordering):
        for j,name2 in enumerate(node_ordering):
            S_reordered[i,j] = S[native_ordering.index(name1),
                                 native_ordering.index(name2)]
    return S_reordered

def prior_layout(node):
    if 'prior' in node.features:
        C = ete3.CircleFace(
            radius=20*np.sqrt(node.prior),
            color="RoyalBlue",
            style="circle",
            #label={'text': "%.5f" % node.prior,
            #       'fontsize': 5}
            )
        C.opacity = 0.5
        ete3.faces.add_face_to_node(C,node,0,position="float")

def prior_style(tree):
    ts = ete3.TreeStyle()
    ts.layout_fn = prior_layout
    ts.show_leaf_name = True
#    ts.mode = "c"
    # tree.show(tree_style=ts)
    nstyle = ete3.NodeStyle()
    nstyle['size'] = 0
    for n in tree.traverse():
        n.set_style(nstyle)
    return ts


