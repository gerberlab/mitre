""" Utilities for transforming data as Dataset objects.

"""
import numpy as np
import pandas as pd
from ..rules import Dataset

def exponentiate(data):
    new_data = data.copy()
    for i in xrange(len(new_data.X)):
        new_data.X[i] = np.exp(data.X[i])
    return new_data

def log_transform(data,zero_data_offset=1e-6,zero_tolerance=1e-10):
    new_data = data.copy()
    # We expect the data to be positive, so don't take the absolute
    # value before checking to see what is close to zero
    for i in xrange(len(new_data.X)):
        new_data.X[i][new_data.X[i]<zero_tolerance] = zero_data_offset
        new_data.X[i] = np.log(new_data.X[i])
    return new_data

def take_relative_abundance(data):
    """ Transform abundance measurements to relative abundance. """

    new_data = data.copy()
    n_subjects = len(new_data.X)
    for i in xrange(n_subjects):
        abundances = new_data.X[i]
        # Data may be integer (counts): cast carefully
        # to float
        total_abundances = np.sum(abundances, axis=0).astype(np.float)
        relative_abundances = abundances/total_abundances
        new_data.X[i] = relative_abundances
    return new_data

def aggregate_on_tree(data, tree_of_variables):
    """ Aggregate data for internal nodes of a tree.

    Arguments:
    data - a rules.Dataset object
    tree_of_variables: a Tree whose leaf names match the variable
    names in data. Leaf names with no corresponding variable will be 
    treated as if the corresponding data is uniformly 0. Variable 
    names with no corresponding leaf will be left unmodified. 

    Returns:
    A new Dataset, adding variables corresponding to internal nodes
    of the tree, with data corresponding to the sum of the data
    for all the descendant leaves of those nodes in each sample.

    The variable_weights of the input data is ignored. The variable_weights
    of the output data is set to a uniform value.

    """
    old_variables_to_indices = {v: i for i,v in enumerate(data.variable_names)}
    new_variables_to_old_indices = {}
    n_new_variables = len(tree_of_variables.get_descendants()) # excludes the root

    new_variables = []
    for node in tree_of_variables.iter_descendants():
        new_variables_to_old_indices[node.name] = [
            old_variables_to_indices[l.name] for 
            l in node.get_leaves() if 
            l.name in old_variables_to_indices
        ]
        new_variables.append(node.name)
    matched_variables = set(new_variables)
    for v,i in old_variables_to_indices.iteritems():
        if v not in matched_variables:
            new_variables.append(v)
            new_variables_to_old_indices[v] = [i]
    n_new_variables = len(new_variables)

    new_variable_weights = np.ones(len(new_variables))/len(new_variables)
    new_X = []
    for subject in data.X:
        (_,nt) = subject.shape
        new_data = np.zeros((n_new_variables, nt))
        for i,new_variable in enumerate(new_variables):
            indices = new_variables_to_old_indices[new_variable]
            new_data[i,:] = np.sum(subject[indices],0)
        new_X.append(new_data)
    
    return Dataset(new_X, data.T, data.y, 
                   new_variables, new_variable_weights,
                   data.experiment_start, data.experiment_end,
                   variable_annotations = data.variable_annotations.copy())
    
def select_variables(dataset, keep_variable_indices):
    """ Copy the dataset, retaining only specified 
    variables. 

    Raises ValueError if keep_variable_indices is empty.

    Note that, if dataset has a variable_tree attribute, 
    the tree will be pruned to keep only those nodes which are
    kept variables, and the additional nodes required to preserve the
    topology of the tree connecting them; thus, not all nodes in the 
    resulting variable_tree are guaranteed to be variables. 

    """
    if not keep_variable_indices:
       raise ValueError('No variables to be kept.')

    new_variable_names = [dataset.variable_names[i] for
                          i in keep_variable_indices]
    # We want to index into copies of the arrays in dataset.X
    # so that the underlying data is copied instead of referenced.
    temp_dataset = dataset.copy()
    new_X = []
    for subject_X in temp_dataset.X:
        if len(subject_X) == 0:
            new_X.append(subject_X)
        else:
            new_X.append(subject_X[keep_variable_indices])
    new_variable_weights = (
        temp_dataset.variable_weights[keep_variable_indices]
    )
    # This seems a bit redundant but leaves open the possibility
    # of changes to the internals of the Dataset class, 
    # ensures that the n_variables attribute is updated, etc.
    # TODO: make this a copy operation and then update only
    # the needed attributes...
    new_dataset = Dataset(
        new_X, temp_dataset.T, temp_dataset.y,
        new_variable_names, new_variable_weights,
        temp_dataset.experiment_start,
        temp_dataset.experiment_end,
        subject_IDs=temp_dataset.subject_IDs,
        subject_data=temp_dataset.subject_data,
        additional_subject_categorical_covariates=temp_dataset.additional_subject_categorical_covariates,
        additional_covariate_default_states=temp_dataset.additional_covariate_default_states,
        additional_subject_continuous_covariates=temp_dataset.additional_subject_continuous_covariates,
        variable_annotations = temp_dataset.variable_annotations.copy()
    )
    if hasattr(temp_dataset, 'variable_tree'):
        new_tree = temp_dataset.variable_tree.copy()
        old_node_names = {n.name for n in new_tree.get_descendants()}
        new_nodes = [v for v in new_variable_names if v in old_node_names]
#        print 'debug select variables: new nodes:'
#        print new_nodes
        if new_nodes:
            new_tree.prune(new_nodes, preserve_branch_length=True)
            new_dataset.variable_tree = new_tree
        # Otherwise there is no point in retaining the tree as we
        # have dropped all variables with a tree relationship.
    return new_dataset

def select_subjects(dataset, keep_subject_indices, invert=False):
    """ Copy the dataset, retaining only specified 
    subjects. 
    
    Raises ValueError if keep_subject_indices is empty.

    If invert is True, keep all subjects _except_ those specified.

    dataset - rules.Dataset instance
    keep_subject_indices - list or array of numbers, the indices (into
    dataset.X/dataset.T/dataset.subject_IDs) of the subjects to be
    retained.

    """
    if len(keep_subject_indices) < 1:
       raise ValueError('No subjects to be kept.')

    if invert:
        exclude_indices = set(keep_subject_indices)
        keep_subject_indices = [i for i in xrange(dataset.n_subjects) if
                                i not in exclude_indices]
    new_data = dataset.copy()
    if new_data.additional_covariate_matrix is not None:
        new_data.additional_covariate_matrix = dataset.additional_covariate_matrix[keep_subject_indices]
    new_X = []
    new_T = []
    new_y = []
    new_subject_IDs = []
    for i in keep_subject_indices:
        new_X.append(new_data.X[i])
        new_T.append(new_data.T[i])
        new_y.append(new_data.y[i])
        new_subject_IDs.append(new_data.subject_IDs[i])
    new_data.X = new_X
    new_data.T = new_T
    new_data.y = np.array(new_y)
    new_data.subject_IDs = new_subject_IDs
    new_data.n_subjects = len(new_subject_IDs)
    if isinstance(new_data.subject_data, pd.DataFrame):
        new_data.subject_data = new_data.subject_data.loc[new_subject_IDs]
    new_data._primitive_result_cache = {}
    return new_data

def make_sister_leaves_mutually_exclusive(data, tree_of_variables):
    """ Simulate mutual exclusion of closely related taxa.

    This is a crude approach which simply identifies each pair of
    sibling leaves on the tree (leaves whose immediate sibling nodes
    are not also leaves are ignored), and for each subject, chooses
    one of the pair at random and sets its corresponding data to zero. 
    
    Raises ValueError if any such leaf is not among the variables in
    the dataset, or if the tree is not binary.

    Arguments:
    data - Dataset to which the operation should be applied.
    tree_of_variables - Phylogenetic tree of a subset of the 
    variables in data. Binary.

    Returns:
    A new Dataset.

    """
    new_data = data.copy()
    leaves = tree_of_variables.get_leaves()
    leaf_sister_pairs = []
    while leaves:
        leaf1 = leaves.pop()
        sisters = leaf1.get_sisters()
        if len(sisters) != 1:
            raise ValueError('Malformed tree.')
        leaf2 = sisters[0]
        if not leaf2.is_leaf():
            continue
        leaves.remove(leaf2)
        leaf_sister_pairs.append((leaf1.name, leaf2.name))
    leaves_to_indices = {name: data.variable_names.index(name)
                         for pair in leaf_sister_pairs for 
                         name in pair}
    for i in xrange(new_data.n_subjects):
        for pair in leaf_sister_pairs:
            drop_variable = np.random.choice(pair)
            drop_index = leaves_to_indices[drop_variable]
            new_data.X[i][drop_index] = 0.
    return new_data
        
        
    
