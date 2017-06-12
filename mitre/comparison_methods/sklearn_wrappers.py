""" Fit alternative models to data given as Dataset objects.

We use the scikit-learn implementations of random forests and ordinary
logistic regression with L1 regularization and cross-validation of the
regulatization parameter.

"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV

class Wrapper():
    """ Base class for sklearn wrappers.

    """
    def __init__(self, data, N_i, N_c):
        """ Prepare to fit a model to a Dataset object.
        
        """
        self.N_i = N_i
        self.N_c = N_c
        self.fit_X = self.transform_X(data)
        # Multiply y by 1 to cast booleans to int if needed
        self.fit_y = 1*data.y
        self.variable_ordering = data.variable_names[:]
        
    def transform_X(self, data, N_i=None, N_c=None):
        """ Average and reformat data.
        
        The experimental interval is divided into time windows and the
        data, presumed to be irregularly sampled, are transformed into
        a regular format by taking the average value of each variable
        for each subject in each time window. 

        Arguments: 
        data - Dataset object
        N_i, N_c - parameters defining allowed time windows. If None
        (the default), the values specified when this object was
        initialized are used. The specification of time windows
        proceeds as follows: the time from the start of the experiment
        to its end, as defined in the Dataset object, is divided up
        into N_i equal intervals; then each sequence of exactly N_c
        consecutive such intervals is a valid time window. (Thus, time
        windows may overlap.) This lets us apply these classifiers to
        the same datasets as the rule-based models, for which we
        preprocess data by discarding subjects which do not possess a
        minimum number of observations within a set of time windows
        defined similarly.
        
        Returns: an array containing the averaged values, of shape
        n_subjects x (n_variables * n_windows).

        """
        if N_i is None:
            N_i = self.N_i
        if N_c is None:
            N_c = self.N_c

        interval_edges = np.linspace(data.experiment_start, data.experiment_end, N_i + 1)
        windows = zip(interval_edges, interval_edges[N_c:])
        new_X = np.zeros((data.n_subjects, len(windows)*data.n_variables))
        # Code for averaging within a window needs to be consistent with the 
        # evaluate method of PrimitiveRule objects in rules.py, and is somewhat 
        # duplicative of it; this should be addressed in later releases.
        for subject_index in xrange(data.n_subjects):
            column_index = 0
            subject_data = data.X[subject_index]
            subject_timepoints = data.T[subject_index]
            for t0, t1 in windows:
                relevant_time_indices = (subject_timepoints >= t0) & (subject_timepoints <= t1)
                relevant_data = subject_data[:,relevant_time_indices]
                if len(relevant_data) == 0:
                    raise ValueError('Cannot classify subject %d: no points within time window %.3f-%.3f.' %
                                     (subject_index, t0, t1))
                for variable_index in xrange(data.n_variables):
                    average = np.mean(relevant_data[variable_index])
                    new_X[subject_index,column_index] = average
                    column_index += 1
        return new_X

    def predict(self, data):
        if not data.variable_names == self.variable_ordering:
            raise ValueError('Variable ordering mismatch.')
        X = self.transform_X(data)
        return self.classifier.predict(X)

class RandomForestWrapper(Wrapper):
    """ Wraps sklearn's RF class, accounting for differences in data format.

    """

    def __init__(self, data, N_i, N_c, *args, **kwargs):
        """ Fit a random forest model to a Dataset object.
        
        N_i, N_c: parameters defining allowed time windows. See the
        transform_X method.

        args, kwargs: passed to the LogisticRegressionCV constructor.

        """
        Wrapper.__init__(self,data,N_i,N_c)
        kwargs['n_estimators'] = 128
        self.classifier = RandomForestClassifier(*args, **kwargs)
        self.classifier.fit(self.fit_X,self.fit_y)


class RandomForestWrapper1K(Wrapper):
    """ Wraps sklearn's RF class, accounting for differences in data format.

    """

    def __init__(self, data, N_i, N_c, *args, **kwargs):
        """ Fit a random forest model to a Dataset object.
        
        N_i, N_c: parameters defining allowed time windows. See the
        transform_X method.

        args, kwargs: passed to the LogisticRegressionCV constructor.

        """
        Wrapper.__init__(self,data,N_i,N_c)
        kwargs['n_estimators'] = 1024
        self.classifier = RandomForestClassifier(*args, **kwargs)
        self.classifier.fit(self.fit_X,self.fit_y)


class RandomForestWrapper32K(Wrapper):
    """ Wraps sklearn's RF class, accounting for differences in data format.

    """

    def __init__(self, data, N_i, N_c, *args, **kwargs):
        """ Fit a random forest model to a Dataset object.
        
        N_i, N_c: parameters defining allowed time windows. See the
        transform_X method.

        args, kwargs: passed to the LogisticRegressionCV constructor.

        """
        Wrapper.__init__(self,data,N_i,N_c)
        kwargs['n_estimators'] = 32768
        self.classifier = RandomForestClassifier(*args, **kwargs)
        self.classifier.fit(self.fit_X,self.fit_y)

class L1LogisticRegressionWrapper(Wrapper):
    """ Wraps sklearn's LogisticRegressionCV class, munging data as needed.

    Despite the name, can be used with other regularization choices as well.

    """

    def __init__(self, data, N_i, N_c, *args, **kwargs):
        """ Fit a regularized logistic regression model to a Dataset object.
        
        By default, uses L1 regularization with the strength chosen from
        10 options spaced logarithmically between 1e-4 and 1e4 
        (the sklearn LogisticRegressionCV default) using
        min(10,data.n_subjects) folds of crossvalidation, but other
        options may be chosen by specifing arguments to the 
        LogisticRegressionCV constructor through *args and **kwargs.
        
        N_i, N_c: parameters defining allowed time windows. See the
        transform_X method.

        args, kwargs: passed to the LogisticRegressionCV constructor.

        """
        Wrapper.__init__(self,data,N_i,N_c)
        default_folds = min(10,data.n_subjects)
        default_classifier_arguments = {
            'cv': default_folds,
            'solver': 'liblinear',
            'penalty': 'l1', 
        }
        # Update with the arguments passed in by the user, clobbering
        # the default settings if alternate values are provided.
        default_classifier_arguments.update(kwargs)
        self.classifier = LogisticRegressionCV(
            *args, 
            **default_classifier_arguments
        )
        self.classifier.fit(self.fit_X,self.fit_y)
