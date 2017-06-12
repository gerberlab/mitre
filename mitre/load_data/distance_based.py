"""
Distance-based clustering of observed sequences. 

Forms a tree using hierachical clustering, calibrates it 
based on phylogenetic information, and aggregates data for higher nodes.

"""
# next: build ete3 tree
# aggregate on ete3
# calibrate phylogeny on tree
# user-friendly descriptions of internal nodes of this tree
import numpy as np
from Bio import SeqIO, pairwise2
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
import pandas as pd

#import ete3

def tree_from_linkage_matrix(linkage, leaf_labels):
    """ Form an ete3.Tree from hierarchical linkage matrix.

    Linkage should be the matrix returned by hierarchy.linkage. 
    leaf_labels should be a vector of names for the nodes 
    corresponding to the clustered items. Internal nodes will be 
    named node0, node1, etc, in the order in which the 
    clusters they represent were formed. 

    returns: new Tree

    """
    
    

def load_dada2_placements(placement_filename, sequence_filename):
    """ Import table of placements, rename according sequence_filename.

    """
    df = pd.read_csv('karelia_test/dada2_placements.csv',index_col=0)
    seq_names = {v:k for k,v in fasta_to_dict(sequence_filename).iteritems()}
    print seq_names.items()[:5]
    df.rename(seq_names)
    return df

def fasta_to_dict(fasta_filename):
    # not clear the use of biopython gains anything here
    with open(fasta_filename) as f:
        return {k: str(v.seq) for k,v in 
                SeqIO.to_dict(SeqIO.parse(f,'fasta')).iteritems()}

def cluster(target_sequence_ids, fasta_filename, method='average'):
    """ Form distance-based hierachical clustering of sequences.

    Looks up each entry in target_sequence_ids in the file 
    specified by fasta_filename to obtain an associated DNA 
    sequence. 
    
    In principle, we could just work with the Hamming distance, but 
    the sequences may be of different lengths (mostly small 
    differences.) So we need a more sophisticated approach: we use
    pairwise global alignment, scoring 0 for a match, -1 for mismatch,
    and -1.5 for opening or extending a gap. We then take the distance
    to be -1.0*(score). 

    UPGMA clustering is used when method='average', the default.

    Returns the distance matrix and the linkage matrix returned
    by the clustering routine.

    """
    # globalms arguments: seq1, seq2, match, mismatch, open, extend
    distance = lambda seq1, seq2: -1.0*(
        pairwise2.align.globalms(seq1,seq2,0,-1,-1.5,-1.5, score_only=True)
    )
    sequences = fasta_to_dict(fasta_filename)
    N = len(target_sequence_ids)
    distances = np.zeros((N,N))
    # fill in the upper triangle
    for i,seqid1 in enumerate(target_sequence_ids):
        seq1 = sequences[seqid1]
        for j_offset, seqid2 in enumerate(target_sequence_ids[i+1:]):
            j = j_offset + i + 1
            seq2 = sequences[seqid2]
            distances[i][j] = distance(seq1, seq2)
    # convert to the form expected by the scipy clustering routines
    y = squareform(distances,checks=False)
    return distances, hierarchy.linkage(y,method)

