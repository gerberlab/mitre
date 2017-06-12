""" 
Code to generate predictive models of other types for comparison purposes.

"""
from sklearn.metrics import roc_auc_score, confusion_matrix
from .sklearn_wrappers import L1LogisticRegressionWrapper as L1LogisticRegression
from .sklearn_wrappers import RandomForestWrapper1K as RandomForest
from .sklearn_wrappers import RandomForestWrapper1K as RandomForest1K
from .sklearn_wrappers import RandomForestWrapper32K as RandomForest32K


