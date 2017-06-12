import pandas as pd
from mitre.load_data import pplacer
from mitre.load_data.basic import fasta_to_dict, dequote
from ..rules import logger

def annotate_dataset_pplacer(dataset,         
                             jplace_filename,
                             taxa_table_filename,
                             seq_info_filename):
    """ Add taxonomic information to dataset (from pplacer info).

    Here we use the pplacer reference package to label inner nodes of
    the reference tree, then we label leaves (OTUs, etc) according to
    their parent nodes.

    Note this should be applied at a stage in the filtering
    process when all leaves of the trees correspond to OTUs/RSVs 
    in the original observed data, and vice versa- otherwise
    higher groups will be misidentified as OTUs.

    Annotations are stored in dataset.variable_annotations, a
    dictionary keyed by variable name. It is not updated as the
    dataset is transformed, dropping variables.  (As always, it's the
    user's responsibility to ensure names are unique.)

    dataset: rules.Dataset object
    jplace_filename: pplacer output .jplace file 
    taxa_table_filename: pplacer refpkg taxa table
    seq_info_filename: pplacer refpkg seqinfo file

    Returns: none.

    """
    node_labels = pplacer.describe_tree_nodes_with_taxonomy(
        jplace_filename,
        taxa_table_filename,
        seq_info_filename
    )
    # We're going to traverse this tree with no particular
    # regard for inefficiency.
    annotations = {}
    nodes = list(dataset.variable_tree.iter_descendants())
    nodes.append(dataset.variable_tree)
    for node in nodes:
        if node.is_leaf():
            parent = node.up.name
            parent_clade = node_labels[parent]
            otu_string = 'OTU mapped to %s' % parent_clade
            annotations[node.name] = otu_string
        else:
            n_leaves = len(node)
            clade = node_labels[node.name]
            annotations[node.name] = (
                '%s' % 
                (clade,)
            )
    dataset.variable_annotations = annotations


def annotate_dataset_hybrid(dataset,         
                            jplace_filename,
                            taxa_table_filename,
                            seq_info_filename,
                            placement_filename,
                            sequence_key_fasta_filename=None
                            ):
    """ Add taxonomic information to dataset (from pplacer info plus table).

    Here we use the pplacer reference package to label inner nodes of
    the reference tree. Leaves are labeled according to a table (eg,
    from mothur or dada2's RDP-based or
    RDP-plus-exact-sequence-match-based taxonomic placement utilities)
    if the table contains a placement of the leaf to the species
    level; otherwise, they are labeled according to their parent node
    in the pplacer results.

    In practice, we have found that the dada2 RDP plus sequence
    matching approach often provides more specific placements for many
    OTUs/RSVs than the pplacer approach. This method lets us take
    advantage of that to obtain good annotations while also continuing
    to define inner nodes based on the pplacer reference tree.

    Note this should be applied at a stage in the filtering
    process when all leaves of the trees correspond to OTUs/RSVs 
    in the original observed data, and vice versa- otherwise
    higher groups will be misidentified as OTUs.

    Annotations are stored in dataset.variable_annotations, a
    dictionary keyed by variable name. It is not updated as the
    dataset is transformed, dropping variables.  (As always, it's the
    user's responsibility to ensure names are unique.)

    dataset: rules.Dataset object
    jplace_filename: pplacer output .jplace file 
    taxa_table_filename: pplacer refpkg taxa table
    seq_info_filename: pplacer refpkg seqinfo file
    placement_filename: csv table of OTU/RSV taxonomies
    sequence_key_fasta_filename: optional fasta file; if given, the
    row labels in the placement file are presumed to be DNA sequences,
    as in DADA2 output, and mapped back to the ids in the fasta where
    possible before return. By default all keys are stripped of
    quotation marks.

    Returns: none.

    """
    # First label everything using the pplacer tree.
    # For efficiency, specify which labels we are looking for.
    nodes = list(dataset.variable_tree.iter_descendants())
    nodes.append(dataset.variable_tree)
    node_label_targets = [node.name for node in nodes if not node.is_leaf()]
    
    node_labels = pplacer.describe_tree_nodes_with_taxonomy(
        jplace_filename,
        taxa_table_filename,
        seq_info_filename,
        to_label = node_label_targets
    )
    # We're going to traverse this tree with no particular
    # regard for inefficiency.
    annotations = {}
    for node in nodes:
        if node.is_leaf():
            parent = node.up.name
            parent_clade = node_labels[parent]
            otu_string = 'OTU mapped to %s' % parent_clade
            annotations[node.name] = otu_string
        else:
            n_leaves = len(node)
            clade = node_labels[node.name]
            annotations[node.name] = (
                '%s' % 
                (clade,)
            )
    # Now read in the table and replace labels
    # for the leaves which are assigned to the species level within the table.

    # First get the loading of the sequence key out of the way,
    # where necessary.
    # Recall the RSVs are by definition unique
    sequence_id_map = {}
    if sequence_key_fasta_filename is not None:
        sequence_id_map.update(
             fasta_to_dict(sequence_key_fasta_filename).iteritems()
        )

    # Load the placement table
    placements = pd.read_csv(placement_filename, index_col=0)
    placements = placements.rename(index=sequence_id_map)
    
    # Grab species-specific OTU labels
    otu_labels = {}
    for id_, record in placements.iterrows():
        if record.notnull()['Species']:
            label = '%s %s' % (record['Genus'], record['Species'])
            otu_labels[id_] = 'OTU mapped to %s' % label
    annotations.update(otu_labels)

    # Finally, save the annotations.
    dataset.variable_annotations = annotations


def annotate_dataset_table(dataset, 
                           placement_filename,
                           sequence_key_fasta_filename=None):
    """ Map sequence IDs to sensible taxonomic labels (from table.)

    Here we work with a table of the format typically produced 
    from dada2's RDP-based or RDP-plus-exact-sequence-match-based
    taxonomic placement utilities, specifically a CSV table
    mapping OTUs/RSVs to taxonomic groupings at various levels.

    From this table we assign a succinct description to each 
    OTU/RSV, and describe each higher node in the tree based 
    on the taxonomic groupings of the OTUs which descend from it.

    Note this should be applied at a stage in the filtering
    process when all leaves of the trees correspond to OTUs/RSVs 
    in the original observed data, and vice versa- otherwise
    higher groups will be misidentified as OTUs.

    Annotations are stored in dataset.variable_annotations, a
    dictionary keyed by variable name. It is not updated as the
    dataset is transformed, dropping variables.  (As always, it's the
    user's responsibility to ensure names are unique.)

    dataset: rules.Dataset object 
    node_labels: dictionary mapping node
    placement_filename: csv table of OTU/RSV taxonomies
    sequence_key_fasta_filename: optional fasta file; if given, the
    row labels in the placement file are presumed to be DNA sequences,
    as in DADA2 output, and mapped back to the ids in the fasta where
    possible before return. By default all keys are stripped of
    quotation marks.

    Returns: none.

    """
    # First get the loading of the sequence key out of the way,
    # where necessary.
    # Recall the RSVs are by definition unique
    sequence_id_map = {}
    if sequence_key_fasta_filename is not None:
        sequence_id_map.update(
             fasta_to_dict(sequence_key_fasta_filename).iteritems()
        )

    # Load the placement table
    placements = pd.read_csv(placement_filename, index_col=0)
    placements = placements.rename(index=sequence_id_map)
    
    # Process to label the OTUs
    otu_labels = {}
    for id_, record in placements.iterrows():
        if record.notnull()['Species']:
            label = '%s %s' % (record['Genus'], record['Species'])
        elif record.notnull().any():
            placed_levels = record[record.notnull()]
            label = '%s %s' % (placed_levels.index[-1],
                               placed_levels.values[-1])
        else:
            label = '(unclassifiable sequence)'
        otu_labels[id_] = 'OTU mapped to %s' % label

    # Process to label the nodes
    nodes = list(dataset.variable_tree.iter_descendants())
    nodes.append(dataset.variable_tree)
    node_labels = {}
    for node in nodes:
        if node.is_leaf():
            continue

        n_leaves = len(node)
        leaf_names = node.get_leaf_names()
        subtable = placements.loc[leaf_names]
        taxa = []
        for level_index, level in enumerate(subtable.columns):
            valid = subtable[level].notnull()
            values = list(set(subtable[level][valid].values))
            if subtable[level].isnull().any():
                values.append('(unclassifiable)')
            taxa.append(values)
            if (len(values) > 1) or (subtable[level].isnull().any()):
                break

        # Case 1: Single species, or species indistinguishable
        if (level.lower() == 'species' and len(taxa[-1]) == 1):
            descriptor = taxa[-2][0] + ' ' + taxa[-1][0]
        # Case 2: At least one consensus level
        # (before a non-consensus level)
        elif len(taxa) > 1:
            consensus_level = subtable.columns[level_index-1].lower()
            split_level = subtable.columns[level_index].lower()
            consensus_value = taxa[level_index-1][0]
            split_values = ', '.join(map(str,taxa[-1]))
            descriptor = ('a clade within %s %s, '
                          'including representatives of %s %s' %
                          (consensus_level, consensus_value,
                           split_level, split_values))
        # Case 3: No consensus levels
        else:
            split_level = subtable.columns[level_index].lower()
            split_values = ', '.join(map(str,taxa[-1]))
            descriptor = ('a clade including representatives of %s %s' %
                          (split_level, split_values))
        node_labels[node.name] = (
            '%s (with %s OTUs [before filtering])' % 
            (descriptor, n_leaves)
        )
    annotations = node_labels.copy()
    annotations.update(otu_labels)
    dataset.variable_annotations = annotations
