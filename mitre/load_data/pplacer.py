"""Load a pplacer result from .jplace file, extract useful information.

The relevant tree format is described in 

"A FORMAT FOR PHYLOGENETIC PLACEMENTS"
Frederick A Matsen, Noah G Hoffman, Aaron Gallagher, and 
Alexandros Stamatakis, arXiv:1201.3397v1 [q-bio.PE],

section 4.1:
--
"To represent the tree, we extend the well-known Newick file
format. In that format, commas and parentheses are used to display the
structure of the tree. The taxon names (leaf labels) are inserted as
plain text. It is also common to label internal nodes with strings
appearing after the closing of a parenthesis. It is also possible to
label edges of the tree with strings enclosed in square brackets. For
example, the tree

((A:.01[e], B:.01)D:.01[g], C:.01[h]);

is a tree with some edge labels and some node labels.  We extend this
format with edge numberings in curly braces:

((A:.01[e]{0}, B:.01{1})D:.01{3}[g], C:.01{4}[h]){5};

These edge numberings index the edges for the placements. We use curly
braces to distinguish between our edge numberings and other edge label
values such as posterior probability or bootstrap branch (bipartition)
support."
--

Unfortunately neither format is natively quite acceptable to ete3.

For the reference package I am using, anyway, we obtain neither
internal node names nor square-bracketed edge labels, just support
values (?), distances, and edge numbers:

"(...)1.000:1.10024{16223}..."

On the plus side, the edge numbers provide a unique set of internal
node identifiers, but there is the problem of the leaves:

"(S000133633:0.04147{16238},S000388827:0.03647{16239})"

We do want to keep some connection to the original leaf identifiers,
but we are primarily interested in the edges, 

If I could rewrite this I would create a PplacerResult class and
construct an object for each read placement, but this is not 
a high priority...

"""

import json, re
import numpy as np
import pandas as pd
import ete3
from ete3 import Tree
from ..rules import Dataset
from ..rules import logger

# Example Newick strings for debugging.
standard = "((A:.01[e], B:.01)D:.01[g], C:.01[h]);"
edged = "((A:.01[e]{0}, B:.01{1})D:.01{3}[g], C:.01{4}[h]){5};"
typical = "((A:.01{0}, B:.01{1})1.0:.01{3}, C:.01{4}){5};"

edge_number_pattern = re.compile(
    r'([^{}():,]+):([^{}():,]+)\{([^{}():,]+)\}'
)
terminal_edge_number_pattern = re.compile(r'\{([^{}():,]+)\};$')

def load_jplace(filename):
    with open(filename) as f:
        jplace = json.load(f)
    return jplace

def reformat_tree(jplace):
    """ Convert edge-numbered Newick to (more) conventional Newick.

    We expect an input tree of the form:

    ((A:.01{0}, B:.01{1})1.0:.01{3}, C:.01{4}){5};

    (in jplace['tree']- jplace should be the output of load_jplace)
    and wish to obtain a tree with unique internal node identifiers,
    but without edge numbering. We also don't care about internal node
    support values.
    
    We achieve this by reidentifying each node (internal and leaf) 
    with its edge number (that is, the edge number of the edge 
    connecting it to its parent:)

    (("0":.01, "1":.01)"3":.01, "4":.01)"5"; 

    The resulting Newick string is read into an ete3.Tree and that is 
    returned as the first value.

    We also keep track of the original names (for leaves) or 
    support values (for internal nodes) and return two dictionaries:
    one mapping original leaf names to new ids, and one mapping
    new ids to collections of the original names of all descendant
    leaves. Note the edge numbers appear in these dictionaries as strings,
    whether they are keys or values, eg {'0': A, ...}. 

    """
    tree_string = jplace['tree']
    new_id_to_old_value = {}
    for match in edge_number_pattern.finditer(tree_string):
        value, distance, edge_number = match.groups()
        new_id_to_old_value[edge_number] = value

    relabel = lambda match: match.group(3) + ':' + match.group(2)

    new_tree, _ = edge_number_pattern.subn(relabel,tree_string)
    new_tree = terminal_edge_number_pattern.sub(
        lambda match: '%s;' % match.group(1),new_tree
    )

    new_tree = Tree(new_tree, format=1)
    
    leaf_to_new_id = {}
    new_id_to_old_leaf_names = {} 

    for node in new_tree.traverse(strategy='levelorder'):
        name = node.name
        if node.is_leaf():
            leaf_to_new_id[new_id_to_old_value[name]] = name
        old_names_of_leaves = []
        for leaf in node.get_leaves():
            old_names_of_leaves.append(
                new_id_to_old_value[leaf.name]
            )
        new_id_to_old_leaf_names[name] = old_names_of_leaves
        
    return new_tree, leaf_to_new_id, new_id_to_old_leaf_names

def organize_placements(jplace_as_dict):
    """ Extract simple list of placement candidates from jplace import.

    Input should be the result of load_jplace.

    Output: 

    {'placed_sequence_name_0': [(likelihood_weight_ratio_01, edge_number_01, distal01, pendant01),
                                (likelihood_weight_ratio_02, edge_number_02, distal02, pendant02), 
                                ...],
     ...}

    Edge numbers, which in the current somewhat recondite scheme become
    node names, are converted to strings.

    """
    placements = {}
    fields = jplace_as_dict['fields']
    for entry in jplace_as_dict['placements']:
        keys = [l[0] for l in entry['nm']]
        this_sequence_placements = []
        for record in entry['p']:
            attributes = dict(zip(fields,record))
            this_sequence_placements.append(
                (attributes['like_weight_ratio'],
                 str(attributes['edge_num']),
                 attributes['distal_length'],
                 attributes['pendant_length'])
            )
        this_sequence_placements.sort()
        for key in keys:
            placements[key] = this_sequence_placements[:]
    return placements
                              
def extract_weights(tree, placements, target_sequences, prune=True):
    """ Identify OTU ancestors, subtree weights.

    Arguments:
    tree - (first) return value of reformat_tree
    placements - return value of organize_placements
    target_sequences - list of sequences whose placement on the 
    tree is of interest (should match placed sequence identifiers
    in the original jplace file) 

    If prune is True, the reference tree is pruned, retaining only
    those nodes/edges to which placements (of target sequences!) are
    made and those needed to maintain topology on the subtree induced
    by the retained nodes (preserving relative distances) before any
    further calculation is done. 

    Returns:
    sequence_to_ancestors - dict mapping each target sequence to 
    a dict mapping node names in the tree to a number.
    This gives (where nonzero) the probability (estimated from the 
    likelihood weight ratio) that each node so indicated (thus, each 
    edge above each such node) is an ancestor of the sequence.

    node_to_weight - dict mapping each node in the tree to the
    estimated length of the subtree descending from the edge
    immediately above that node, and each target sequence to the
    estimated length of the subtree including only the edge connecting
    it to the reference tree. Subtree length estimation proceeds as
    follows:

    tree - the tree used to obtain weights. If prune is True, this is
    a pruned copy of the input tree; otherwise, it is the input tree
    itself.

    - For an ordinary edge of the reference tree, the length includes 
    the edge itself, the sum of the estimated lengths of the edges 
    descending from this edge's terminal node, and the _average_ length
    of all pendant edges placed on this edge (weighted by the estimated
    probability of each such placement.)
    
    - For a target sequence, the length is the expected value, over 
    all placements, of the pendant edge length.

    Note that if prune is true, node_to_weight will contain only
    nodes with a nonzero estimated probability of being an ancestor
    of target_sequences.

    If prune is false and target_sequences=[], node_to_weight
    will give the subtree length associted with each edge in the 
    bare reference tree, ignoring placed sequences.

    """
    sequence_to_ancestors = {}
    node_to_weight = {}
    placements_by_edge = {} # {edge: [(weight1, pendant1, sequence1), ...] ...}
    nonzero_probability_of_ancestry = set()

    target_sequence_set = set(target_sequences)
    if prune:
        # Don't destructively modify the input...
        tree = tree.copy()
        keep_nodes = set()
        keep_nodes = {edge for seq, options in placements.iteritems() for 
                      _, edge, _, _ in options if seq in target_sequence_set}
        tree.prune(list(keep_nodes), preserve_branch_length=True)

    # Rely on the assumption that all nodes have unique names.
    name_to_node = {n.name: n for n in tree.get_descendants()}
    name_to_node[tree.name] = tree
    for sequence in target_sequences:
        ancestors = {}
        for weight, edge, distal, pendant in placements[sequence]:
            # Note the edge _numbers_ from the placements are converted 
            # to node names as _strings_-- this is taken care of in
            # organize_placements.
            placements_by_edge.setdefault(edge,[]).append((weight, pendant, sequence))
            edge_node = name_to_node[edge]
            for node in [edge_node] + edge_node.get_ancestors():
                nonzero_probability_of_ancestry.add(node.name)
                ancestors[node.name] = weight + ancestors.get(node.name,0.)
        sequence_to_ancestors[sequence] = ancestors

    # walk from leaves back up
    # To avoid confusion with the placement weights, refer to these as
    # lengths, even though they are only sort of lengths.
    for node in reversed(list(tree.traverse(strategy='levelorder'))):
        length = node.dist 
        length += sum(node_to_weight[child.name] for child in node.children)
        if node.name in placements_by_edge:
            weights = np.array([weight for weight, _, _ in
                                placements_by_edge[node.name]])
            pendants = np.array([pendant for _, pendant, _ in
                                 placements_by_edge[node.name]])
            weighted_average_pendant = (np.sum(weights*pendants) /
                                        np.sum(weights)) 
            length += weighted_average_pendant
        node_to_weight[node.name] = length

    # Finally handle the placed sequences.
    for sequence, options in placements.iteritems():
        if sequence not in target_sequences:
            continue
        weights, _, _, pendants = zip(*options)
        weights = np.array(weights)
        pendants = np.array(pendants)
        weighted_average_pendant = np.sum(weights*pendants)/np.sum(weights)
        node_to_weight[sequence] = weighted_average_pendant
            
    return sequence_to_ancestors, node_to_weight, tree

def calibrate_pplacer_weights_taxonomically(jplace_filename,
                                            taxa_table_filename,
                                            seq_info_filename, 
                                            must_contain=1.0):
    """Get distribution of subtree lengths across taxonomic levels.

    Only the tree of reference species is considered: placed
    sequences are ignored for this calculation.

    We check all taxa at levels
    ['phylum','class','order','family','genus','species'], excluding
    those where all reference species belong to the same taxon at the
    next level down (except for 'species'.) For each, we find the
    lowest common ancestor on the reference tree, and record the 
    subtree weight associated with the lowest common ancestor.
    (Strictly we are working with the lowest _edge_ that is a 
    common ancestor.) 

    NOT IMPLEMENTED YET: By setting must_contain to a value other than
    1.0 we can effectively exclude 'outliers' which have been placed
    far away from the rest of their taxon (rightly or wrongly!)-- in
    that case we traverse the subtree descended from the LCA to find
    the minimum weight associated with an edge which is ancestral to
    at least ceil(must_contain * number of reference sequences
    assigned to this taxon) of the relevant reference sequences.

    Arguments:

    jplace_filename - pplacer result file
    taxa_table_filename, seq_info_filename - see extract_groups_at_level.
    must_contain - see above

    Returns: 
    dict mapping taxonomic levels to ndarray of the subtree lengths
    associated with taxa at that level.

    """
    jplace = load_jplace(jplace_filename)
    tree, leaf_to_new_id, new_id_to_old_leaf_names = (
        reformat_tree(jplace)
    ) 
    _, bare_reference_weights, _ = extract_weights(
        tree, placements={}, target_sequences=[],
        prune=False
    ) 
    levels = ['phylum' ,'class','order','family','genus','species']
    lower_levels = ['class', 'order','family','genus','species',None]
    results = {}
    for level, check_level in zip(levels, lower_levels):
        # This is somewhat inefficient as extract_groups_at_level
        # rereads the tables and performs the join anew each time,
        # but hopefully we aren't going through this process
        # frequently.
        this_level_subtree_lengths = []
        group_to_leaves = extract_groups_at_level(
            level, taxa_table_filename,
            seq_info_filename, check_level=check_level
        )
        for taxon_name, leaves in group_to_leaves.iteritems():
            new_leaf_ids = [leaf_to_new_id[l] for l in leaves]
            if len(new_leaf_ids) == 1:
                # Note calling tree.get_common_ancestor with one 
                # argument gets the common ancestor of 'tree' itself
                # (here, the root node) and the argument, which 
                # is not what we want.
                lca_name = new_leaf_ids[0]
            else:
                lca_name = tree.get_common_ancestor(*new_leaf_ids).name
            this_level_subtree_lengths.append(
                (taxon_name, lca_name, len(leaves), bare_reference_weights[lca_name])
            )
        results[level] = this_level_subtree_lengths
    return results

def describe_tree_nodes_with_taxonomy(
        jplace_filename,
        taxa_table_filename,
        seq_info_filename,
        to_label=None
        ):

    jplace = load_jplace(jplace_filename)
    tree, leaf_to_new_id, new_id_to_old_leaf_names = (
        reformat_tree(jplace)
    ) 

    _, bare_reference_weights, _ = extract_weights(
        tree, placements={}, target_sequences=[],
        prune=False
    ) 

    taxa_table = pd.read_csv(taxa_table_filename,index_col=0)
    tname = lambda j: str(taxa_table.loc[j]['tax_name'])
    seq_info = pd.read_csv(seq_info_filename,index_col=0)
    reference_species = seq_info.join(taxa_table, on='tax_id', how='inner')

    levels = ['phylum','class','order','family','genus','species']
    results = {}
    
    if to_label is not None:
        to_label = set(to_label)

    for node_id, leaves in new_id_to_old_leaf_names.iteritems():
        if to_label is not None and node_id not in to_label:
            continue

        subtable = reference_species.loc[leaves,:]
        taxa = []
        for level in levels:
            values = list(set(subtable[level].values))
            taxa.append(map(tname,values))
            if len(values) > 1:
                break
        if len(taxa[-1]) == 1:
            descriptor = taxa[-1][0] # ' '.join([l[0] for l in taxa[-2:]])
        elif len(taxa) == 1:
            descriptor = 'a clade within phylum ' + ' or '.join(taxa[0])
        else:
            l = len(taxa) - 2
            descriptor = ('a clade within %s %s,'
                          'including representatives of %s %s' %
                          (levels[l], taxa[l][0], 
                           levels[l+1], ', '.join(taxa[l+1])))
        results[node_id] = descriptor
    return results
    
def extract_groups_at_level(level, taxa_table_filename,
                            seq_info_filename, check_level=None):

    """ Get reference tree leaves in all taxa of a certain level.

    E.g., find every represented genus, and the leaves belonging
    to each of them.

    Arguments: 
    level - e.g. 'genus', 'family'. Should correspond to a column
    heading in taxa_table_filename.
    
    taxa_table_filename - location of table of taxonomic information
    from the pplacer refpkg.

    seq_info_filename - location of reference tree leaf sequence
    information from the pplacer refpkg

    check_level - if given, we skip over taxa at level whose leaves
    all belong to the same taxon at check_level, e.g., we may want to
    exclude families which contain only one genus, because we think
    their subtree length will be more characteristic of genera than
    families.

    Returns a dictionary mapping the name of each (non-skipped)
    taxon at the specified level to the seqname (should be the
    first column of the seq_info file) of each reference sequence
    assigned to that taxon.

    """
    taxa_table = pd.read_csv(taxa_table_filename,index_col=0)
    seq_info = pd.read_csv(seq_info_filename,index_col=0)
    reference_species = seq_info.join(taxa_table, on='tax_id', how='inner')

    results = {}

    for value in set(reference_species[level].dropna().values):
        name = taxa_table.loc[int(value)].tax_name
        leaves = reference_species[reference_species[level]==value]
        if check_level is not None:
            check_values = set(leaves[check_level].dropna().values)
            if len(check_values) <= 1:
                continue
        results[name] = leaves.index.values

    return results
                                        
def aggregate_by_pplacer(jplace_filename, input_dataset):
    """ Aggregate data on a phylogenetic tree using pplacer results. 

    Variable names with no corresponding placements will be left 
    unmodified.

    The variable weights in input_dataset are ignored, 
    and new weights are set according to subtree lengths.

    Returns a new dataset, and a dictionary mapping new variables to 
    subtree lengths. (Old variables that are not subtree leaves 
    are given a weight 1.0 as this is not fully supported yet.)

    Adds the tree of variables as an attribute to the returned dataset.

    """
    jplace = load_jplace(jplace_filename)
    tree, _, _  = reformat_tree(jplace)
    placements = organize_placements(jplace)
    target_sequences = list(
        set(placements).intersection(input_dataset.variable_names)
    )
    sequence_to_ancestors, var_to_weight, pruned_tree = extract_weights(
        tree, placements, target_sequences, prune=True
    )
    
    # Which variables will we be adding, and how many?  Possibly there
    # should have been a return value from extract_weights giving this
    # information-- as it is we have to look at the entries of
    # node_to_weight and exclude those that are already variables
    # (note that there may be existing variables which are not nodes,
    # though there should not be placed sequences which are not
    # variables.)
    added_variables = [v for v in var_to_weight if
                       v not in input_dataset.variable_names]
    old_n_variables = input_dataset.n_variables
    # Note new_n_variables is NOT the number of new variables, and
    # new_variable_names is the new list of names of all the
    # variables, similarly.
    new_n_variables = old_n_variables + len(added_variables) 
    new_variable_names = input_dataset.variable_names + added_variables
    new_variable_indices = dict(
        zip(added_variables,
            old_n_variables + np.arange(len(added_variables))
        )
    )
    new_X = []
    for array in input_dataset.X:
        _, nt = array.shape
        new_array = np.zeros((new_n_variables, nt))
        new_array[:old_n_variables,:] = array
        for i, old_variable in enumerate(input_dataset.variable_names):
            old_observations = array[i]
            for new_variable, weight in sequence_to_ancestors[old_variable].iteritems():
                j = new_variable_indices[new_variable]
                new_array[j,:] += weight * old_observations
        new_X.append(new_array)       
    new_dataset = input_dataset.copy()
    new_dataset.n_variables = new_n_variables
    new_dataset.X = new_X
    new_dataset.variable_names = new_variable_names
    new_dataset.variable_weights = np.array(
        [var_to_weight.get(v,1.0) for v in new_variable_names]
    )
    new_dataset.variable_tree = tree
    return new_dataset, var_to_weight

###

def extract_weights_simplified(tree, placements, target_sequences, prune_before_weighting=True):
    """ Identify OTU ancestors, subtree weights.

    Like extract_weights, but assigns each OTU to a definite best
    ancestor, rather than accounting for all possible placements. 

    See aggregate_by_pplacer_simplified for a description of how
    this is done.

    Arguments:
    tree, placements, target_sequences: See exact_weights.
    
    prune_before_weighting: if True (the default), prune the tree to
    include only OTUs and the higher nodes necessary to preserve
    topological relationships among them _before_ calculating subtree
    weight values. Otherwise, do this afterwards.
    
    Returns:
    sequence_to_ancestors - dict mapping each target sequence to 
    a list of nodes ancestral to its best immediate ancestor, including 
    the best immediate ancestor itself.

    node_to_weight - dict mapping each node in the tree to the
    estimated length of the subtree descending from the edge
    immediately above that node, and each target sequence to the
    estimated length of the subtree including only the edge connecting
    it to the reference tree. N.B.: as in extract_weights, the subtree
    weights assigned to ordinary edges include contributions from
    sequences which may be placed on the subtrees descending from
    those edges. Where above, these contributions were weighted by the
    estimated probability that the sequences are placed there, here we
    instead treat placements as definite. However, subtree weights for
    target sequences do represent a weighted average over all
    placements.

    tree - the tree used to obtain weights, with nodes for OTUs added
    as children of their immediate best ancestor (yes, this is
    conceptually odd as placement is made to edges, not nodes of the
    reference tree, but this tree object describes the hierarchical
    relationship among variables in the model, rather than
    recapitulating a best-estimate phylogeny exactly.) The tree is
    pruned to include only OTUs and the higher nodes necessary to
    preserve the topology.

    node_to_sequences - dict mapping the name of each node to a list 
    of OTUs that descend from it.

    """
    sequence_to_best_ancestors = {}
    sequence_to_all_ancestors = {}
    sequence_typical_pendant_length = {}
    node_to_weight = {}
    node_to_sequences = {}
    # Don't destructively modify the input:
    tree = tree.copy()

    target_sequence_set = set(target_sequences)
    # Rely on the assumption that all nodes have unique names.
    name_to_node = {n.name: n for n in tree.get_descendants()}
    name_to_node[tree.name] = tree

    for sequence in target_sequences:
        # placements[sequence] is a list of tuples, each of the form
        # (likelihood_weight_ratio, edge, distal, pendant) 
        ancestry_probabilities_this_sequence = {}
        weighted_pendant_length = 0
        best_ancestor = False

        for weight, edge, distal, pendant in placements[sequence]:
            weighted_pendant_length += weight * pendant
            ancestor = name_to_node[edge]
            for node in ([ancestor] + ancestor.get_ancestors()):
                ancestry_probabilities_this_sequence.setdefault(node.name, []).append(weight)
            if weight > 0.5:
                best_ancestor = edge
        sequence_typical_pendant_length[sequence] = weighted_pendant_length

        if not best_ancestor:
            sorted_ancestors = sorted(
                [(sum(weights), k) for k, weights in 
                 ancestry_probabilities_this_sequence.iteritems() if
                 sum(weights) > 0.5]
            )
            best_ancestor = sorted_ancestors[0][1]

        sequence_to_best_ancestors[sequence] = best_ancestor
        all_ancestors = (
            [best_ancestor] +
            [n.name for n in name_to_node[best_ancestor].get_ancestors()]
        )
        sequence_to_all_ancestors[sequence] = all_ancestors
        for ancestor in all_ancestors:
            node_to_sequences.setdefault(ancestor,[]).append(sequence)
        
    # Add OTUs as children of best ancestors.
    logger.info('Attaching sequences/OTUs to tree...')
    for otu, best_ancestor in sequence_to_best_ancestors.iteritems():
        dist = sequence_typical_pendant_length[otu]
        best_ancestor_as_node = name_to_node[best_ancestor]
        best_ancestor_as_node.add_child(name=otu,dist=dist)

    if prune_before_weighting: 
        logger.info('Pruning (this may take a moment)...')
        tree.prune(target_sequences, preserve_branch_length=True)

    # walk from leaves back up
    # To avoid confusion with the placement weights, refer to these as
    # lengths, even though they are only sort of lengths.
    logger.info('Calculating weights...')
    for node in reversed(list(tree.traverse(strategy='levelorder'))):
        length = node.dist 
        length += sum(node_to_weight[child.name] for child in node.children)
        node_to_weight[node.name] = length

    if not prune_before_weighting: 
        tree.prune(target_sequences, preserve_branch_length=True)

    # Discard data pertaining to nodes that have been filtered out of the tree
    remaining_nodes = {n.name for n in tree.get_descendants()}
    remaining_nodes.add(tree.name)
    sequence_to_all_ancestors = {s: [ancestor for ancestor in v if ancestor in remaining_nodes] for
                                 s,v in sequence_to_all_ancestors.iteritems()}
    node_to_weight = {k:v for k,v in node_to_weight.iteritems() if k in remaining_nodes}
    node_to_sequences = {k:v for k,v in node_to_sequences.iteritems() if k in remaining_nodes}

    return sequence_to_all_ancestors, node_to_weight, tree, node_to_sequences


def aggregate_by_pplacer_simplified(jplace_filename, input_dataset):
    """ Aggregation on phylogenetic tree using pplacer results (simple)  

    This proceeds much as aggregate_by_pplacer. However, instead of
    tracking all the separate possible placement locations for each
    OTU, each OTU is given a single placement to the lowest node in
    the tree with probability of being an ancestor of the OTU (as
    estimated by summing the LWR of placements to the node's children)
    is greater than 0.5. The OTUs are then added as child nodes of the
    nodes to which they are placed (this is a little odd conceptually,
    as placement is notionally to the edge _above_ the node, but
    accurately reflects the hierarchical relationship we use to
    aggregate data, etc.) The edge length of the edge connecting the
    OTU to its ancestor node is the weighted average pendant length of
    that node's placements (even if the chosen ancestor node should
    correspond to a particular placement with pendand length not
    exactly equal to that average.) Data for the OTU is aggregated to
    the best ancestor and its ancestors in turn as though each had a
    combined LWR=1.0.
    
    Some subtleties: all variable weights are the same as they 
    would be under the more complicated placement process. 

    Note the abundance of nodes lying below a node's best ancestor may
    be a little odd: consider the tree

      /-C
    A-|
      \-B 

    and suppose OTU1 attaches with equal probability to B or C, but
    OTU2 certainly attaches to C, as does OTU3 to B. A will be the
    best ancestor of OTU1, and the data for C will reflect only the
    abundance of OTU2 and B only the abundance of OTU3, even though
    OTU1 certainly must attach to one or the other. We contend this is an
    acceptable approximation: the 'abundance of subtree B [or C]' does
    not exist independent of some defined procedure for mapping reads
    to tree nodes, and this is how we do it. Close inspection of
    results will in any case reveal that a rule attaching to C refers
    to a group of bacteria including OTU2 but not OTU1.

    Return and other behavior the same as for aggregate_by_pplacer.

    """
    logger.info('Loading and reprocessing phylogenetic placements...')
    jplace = load_jplace(jplace_filename)
    tree, _, _  = reformat_tree(jplace)
    placements = organize_placements(jplace)
    target_sequences = list(
        set(placements).intersection(input_dataset.variable_names)
    )

    logger.info('Identifying best placements and calculating subtree weights...')
    result = extract_weights_simplified(
        tree, placements, target_sequences, prune_before_weighting=True
    )
    sequence_to_ancestors, var_to_weight, tree_of_variables, node_to_sequence = result
   
    # Comments on adding the OTUs to the tree.
    # 
    # We need a cleaner way to map nodes back to their descendant
    # input sequences. We also generally want to only assign rules to
    # specific OTUs if the OTU level gives better information than
    # their apparent position on the tree-- if we incorporate the OTUs
    # into the tree, we can achieve this through the tree-based
    # redundancy filtering in cases where an OTU's best ancestor is
    # not the best ancestor of any other OTU suppressed by redundancy
    # filtering. (In cases where a node is the best ancestor of
    # multiple OTUs, we definitely want to preserve the node and the
    # OTUs as separate variables.) So there are good reasons to add
    # each OTU as a child of its immediate best ancestor- it is
    # conceptually a little weird, as in fact the OTU may notionally
    # attach to the reference tree on the edge _above_ the 'ancestor',
    # but we are building a tree that represents a hierarchy of groups
    # of variables, not strictly a phylogeny, for all that the groups
    # and their relationships are almost all phylogenetically defined.
    # (We could alternatively add a new node on the edge above the
    # best ancestor, but then we get into questions of what the
    # appropriate edge length would be in the case where the best
    # ancestor we are confident in is not actually one of the edges to
    # which a possible placement was identified by pplacer.)

    # assume names are unique
    variable_name_to_node = {n.name: n for n in tree_of_variables.get_descendants()}
    variable_name_to_node[tree_of_variables.name] = tree_of_variables

    # Now add variables to the data matrices. We preserve all the old
    # variables in the input dataset, and add the internal nodes of
    # tree_of_variables. Note that here, the data for each node is
    # simply the sum of the data for the OTUs of which it is an
    # ancestor- no need to weight.
    tree_variable_names = [n.name for n in tree_of_variables.get_descendants()]
    # The above doesn't include the root. We don't actually want the
    # root as a variable, but I have assumed it is in other code (and
    # rules applying to it will be filtered out elsewhere.)
    tree_variable_names.append(tree_of_variables.name)
    old_variable_names = set(input_dataset.variable_names)
    added_variables = [v for v in tree_variable_names if
                       v not in old_variable_names]
    old_n_variables = input_dataset.n_variables
    # Note new_n_variables is NOT the number of new variables, and
    # new_variable_names is the new list of names of all the
    # variables, similarly.
    new_n_variables = old_n_variables + len(added_variables) 
    new_variable_names = input_dataset.variable_names + added_variables
    new_variable_indices = dict(
        zip(new_variable_names,
            np.arange(new_n_variables)
        )
    )
    new_X = []
    logger.info('Aggregating data...')
    for array in input_dataset.X:
        _, nt = array.shape
        new_array = np.zeros((new_n_variables, nt))
        new_array[:old_n_variables,:] = array
        ###
        # Here the aggregation code differs from the original function
        for node_name in added_variables:
            descendant_otus = node_to_sequence[node_name]
            node_index = new_variable_indices[node_name]
            for otu in descendant_otus:
                otu_index = new_variable_indices[otu]
                new_array[node_index,:] += new_array[otu_index]
        # 
        ### 
        new_X.append(new_array)   
    logger.info('Finalizing aggregated data...')
    new_dataset = input_dataset.copy()
    new_dataset.n_variables = new_n_variables
    new_dataset.X = new_X
    new_dataset.variable_names = new_variable_names
    new_dataset.variable_weights = np.array(
        [var_to_weight.get(v,1.0) for v in new_variable_names]
    )
    new_dataset.variable_tree = tree_of_variables
    return new_dataset, var_to_weight, node_to_sequence
    
    
