""" Add variables to subject data indicating who has any allergy.

Also, who has any dietary allergy.

Variables will be null/missing for any subject with a missing (relevant)
allergy field.

"""
import pandas as pd
subjects = pd.read_csv('subject_data.csv', index_col='subjectID')
dietary = ['allergy_egg', 'allergy_milk', 'allergy_peanut']
other = ['allergy_%s' % s for s in ['cat', 'dog', 'dustmite',
                                    'birch','timothy']]
dietary_columns = [subjects[s] for s in dietary]
allergy_columns = [subjects[s] for s in dietary + other]

diet_allergies = sum(dietary_columns)
any_allergies = sum(allergy_columns)

diet_allergies[diet_allergies.notnull()] = (
    diet_allergies[diet_allergies.notnull()] > 0 
)
any_allergies[any_allergies.notnull()] = (
    any_allergies[any_allergies.notnull()] > 0 
)

subjects['dietary_allergy'] = diet_allergies
subjects['any_allergy'] = any_allergies

subjects.to_csv('subject_data_augmented.csv',index_label='subjectID')
