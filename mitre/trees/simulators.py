""" Methods for generating simulated phylogenies for testing purposes.

"""

import numpy as np
import ete3
from ete3 import Tree
from ..rules import Dataset

def even_tree(nodes, edge_length=1.0):
    """ Arrange the nodes in a roughly uniform tree, for testing purposes.

    Arguments:
    nodes - list of strings giving the names of the leaves.
    edge_length - length of each edge in the tree (default 1.0)

    """
    working_nodes = [Tree(name=n) for n in nodes]
    internal_node_count = 0
    while len(working_nodes) > 1:
        left = working_nodes.pop(0)
        right = working_nodes.pop(0)
        new = Tree(name='node%d' % internal_node_count)
        new.add_child(left, dist=1)
        new.add_child(right, dist=1)
        internal_node_count += 1
        working_nodes.append(new)
    return working_nodes[0]
        
def uneven_tree(nodes, l1=1, l2=2):
    """ Arrange nodes in a systematically uneven tree. 

    """
    working_nodes = [Tree(name=n) for n in nodes]
    internal_node_count = 0
    while len(working_nodes) > 1:
        left = working_nodes.pop(0)
        right = working_nodes.pop(0)
        new = Tree(name='node%d' % internal_node_count)
        new.add_child(left, dist=l1)
        new.add_child(right, dist=l2)
        internal_node_count += 1
        working_nodes.append(new)
    return working_nodes[0]

def random_tree(nodes, mean_log_distance=0, std_log_distance=1):
    working_nodes = [Tree(name=n) for n in nodes]
    internal_node_count = 0
    while len(working_nodes) > 1:
        left = working_nodes.pop(0)
        right = working_nodes.pop(0)
        new = Tree(name='node%d' % internal_node_count)
        d1, d2 = np.exp(mean_log_distance + std_log_distance*np.random.randn(2))
        new.add_child(left, dist=d1)
        new.add_child(right, dist=d2)
        internal_node_count += 1
        working_nodes.append(new)
    return working_nodes[0]
    

# def treeify(data, tree_of_variables, a_L=4., b_L_scale=2.):
#     """ Aggregate data for internal nodes of a tree.

#     Arguments:
#     data - a rules.Dataset object
#     tree_of_variables: a Tree whose leaf names match the variable
#     names in data
#     a_L, b_L_scale: parameters for the gamma-like prior on rule attachment
    
#     Returns:
#     A new Dataset.

#     """
#     old_variables_to_indices = {v: i for i,v in enumerate(data.variable_names)}
#     new_variables_to_old_indices = {}
#     n_new_variables = len(tree_of_variables.get_descendants()) # excludes the root
#     new_variables = []
#     for node in tree_of_variables.iter_descendants():
#         new_variables_to_old_indices[node.name] = [
#             old_variables_to_indices[l.name] for 
#             l in node.get_leaves()
#         ]
#         new_variables.append(node.name)

# #    new_variable_prior = np.ones(n_new_variables) # temporary
#     new_variable_prior = node_priors(tree_of_variables, a_L, b_L_scale)
#     new_variable_prior = np.array([new_variable_prior[v] for v in new_variables])
#     new_X = []
#     for subject in data.X:
#         (_,nt) = subject.shape
#         new_data = np.zeros((n_new_variables, nt))
# #        print new_data.shape
#         for i,new_variable in enumerate(new_variables):
#             indices = new_variables_to_old_indices[new_variable]
#             new_data[i,:] = np.sum(subject[indices],0)
#         new_X.append(new_data)
    
#     return Dataset(new_X, data.T, data.y, 
#                    new_variables, new_variable_prior,
#                    data.experiment_start, data.experiment_end)
    
