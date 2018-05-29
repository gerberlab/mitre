import pandas as pd
metadata = pd.read_csv('diabimmune_t1d_wgs_metadata.csv')
subject_data = metadata.loc[:,['Subject_ID','Case_Control']]
subject_data = subject_data.drop_duplicates().set_index('Subject_ID')
subject_data.to_csv('t1d_wgs_subject_data.csv')
# It would make sense to use Gid_shotgun, but the Metaphlan table
# matches Gid_16S
sample_metadata = metadata.loc[:,['Gid_16S','Subject_ID','Age_at_Collection']].set_index('Gid_16S')
sample_metadata.to_csv('t1d_sample_metadata.csv',header=False)
