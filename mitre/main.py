""" Handle different operations (preprocessing, model-building, sampling, etc) based on config file.

"""
import sys, pickle, ConfigParser, csv, os, json

# Use a non-interactive matplotlib backend when MITRE is invoked from the command line.
# We need to call this before importing any of the submodules that import matplotlib.
import matplotlib
matplotlib.use('agg')

from load_data import basic, pplacer, taxonomy_annotation
import data_processing.transforms as transforms
import data_processing.filtering as filtering
from data_processing.crossvalidation import stratified_k_folds, leave_one_out_folds, debug_leave_one_out_folds
from mitre import logit_rules, comparison_methods, posterior
import mitre.rules as rules
from scipy.spatial.distance import hamming
from scipy.stats import dirichlet
from sklearn.metrics import roc_auc_score, confusion_matrix
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
import multiprocessing as mp
import logging

logger = logit_rules.logger

comparison_methods_and_labels = [
    ('Random forest', comparison_methods.RandomForest1K),
    # ('Random forest 1028',
    #  comparison_methods.RandomForest1K),
    # ('Random forest 32768',
    #  comparison_methods.RandomForest32K),
    ('L1-regularized logistic regression', comparison_methods.L1LogisticRegression)
]

def run_from_config_file(filename):
    """ Parse a configuration file and carry out operations specified in it.

    """
    config = ConfigParser.ConfigParser()
    config.read(filename)

    current_dataset = None
    current_model = None
    current_sampler = None

    if config.has_section('benchmarking'):
        # Using the 'benchmark' convenience option bypasses the normal
        # flow through the various steps, though note most of the other
        # configuration file sections do still control relevant parts 
        # of the process.
        benchmark(config)
        return 

    if config.has_section('general'):
        if config.has_option('general', 'verbose'):
            if config.getboolean('general','verbose'):
                logger.setLevel(logging.INFO)

    if config.has_section('preprocessing'):
        current_dataset = preprocess(config)

    if config.has_section('model'):
        current_model = setup_model(config, data=current_dataset)

    if config.has_section('sampling'):
        current_sampler = sample(config, model=current_model)

    if config.has_section('postprocessing'):
        current_summary = do_postprocessing(config, sampler=current_sampler)

    if config.has_section('crossvalidation'):
        crossvalidate(config, model=current_model)

    # Leave-one-out crossvalidation is structurally different
    # from the ordinary CV process, and gets its own function.
    if config.has_section('leave_one_out'):
        leave_one_out(config, model=current_model)

    if config.has_section('comparison_methods'):
        do_comparison(config, data=current_dataset)


def benchmark(config):
    """ Test performance of methods on dataset.

    This is a complex chimera of a function which makes sure that the
    comparison methods are applied at the right point in the pipeline
    (currently several points: before phylogenetic aggregation
    (without OTU filtering,) before phylogenetic aggregation (with OTU
    filtering), after phylogenetic aggregation (working with the same
    dataset as MITRE.)) Also it allows, e.g., noise to be added at the
    right point in the pipeline and propagate appropriately to the
    separate comparison method calls as well as the MITRE
    crossvalidation. Either k-fold or leave-one-out crossvalidation may
    be performed.

    How it works:
    
    1. Data is loaded following the options in 'preprocessing'
    normally through the step of conversion to relative abundance. If
    instead the load_data_from_pickle option is supplied in the
    'benchmarking' section, a dataset is loaded from that file, and it
    is assumed that it has been processed through that step
    already. (Note that in that case, jplace_file must still be
    supplied in the 'preprocessing' section, along with all the other
    preprocessing options which apply to steps in the process after
    the conversion to relative abundances.)

    2. If noise_parameter is specificied, noise is added to the
    relative abundances: ie, for each sample, a new vector of relative
    abundances is drawn from the Dirichlet distribution, with
    parameter equal to noise_parameter * (original vector of relative
    abundances.) (Strictly, we replace all values less than the
    parameter zero_threshold with zero_threshold before multiplying by
    noise_parameter.)

    3. The first comparison-methods crossvalidation is run
    immediately, following the options specified in the 'comparison'
    section of the configuration file as usual. (If log-transformation
    is specified, this is applied before the CV is run.) The resulting
    report will be labeled '[...]_benchmark_step3_comparison.txt'

    4. Temporal filtering options are applied to a copy of the dataset
    and a second comparison-methods crossvalidation is run. (If
    log-transformation is specified, this is applied before the CV is
    run.) The resulting report will be labeled
    '[...]_benchmark_step4_comparison.txt'

    5. All later actions in the 'preprocessing' section are applied to
    the dataset normally, such as aggregation on the phylogenetic tree
    and log transformation. A third comparison-methods crossvalidation
    is run. The resulting report will be labeled
    '[...]_benchmark_step5_comparison.txt'
    
    6. The dataset is used for MITRE crossvalidation, following the
    options specified in the 'model' and 'crossvalidation' or 'leave_one_out' 
    section of the configuration file as usual.

    A total of four reports should be written. 
    
    The following options must be specified within the benchmark
    section (if at all):

    load_data_from_pickle (optional)
    noise_parameter

    """
    # STEP 1: PREPROCESSING/LOADING
    if config.has_option('benchmarking','load_data_from_pickle'):
        with open(config.get('benchmarking',
                             'load_data_from_pickle')) as f:
            ra_data = pickle.load(f)  
    else: 
        ra_data = preprocess_step1(config)

    # STEP 2: DO NOISE ADDITION
    if config.has_option('benchmarking','noise_parameter'):
        noise_parameter = config.getfloat(
            'benchmarking',
            'noise_parameter'
        )
        zero_threshold = config.getfloat(
            'benchmarking',
            'zero_threshold'
        )

        new_X = []
        for array in ra_data.X:
            new_samples = []
            # Each column of array is a sample
            for sample in array.T:
                nonzero = sample.copy()
                nonzero[nonzero<zero_threshold] = zero_threshold
                new_samples.append(dirichlet.rvs(nonzero * noise_parameter))
            new_X.append(np.vstack(new_samples).T)
            # unless it's columns?
        ra_data.X = new_X


    # STEP 3: COMPARISON METHODS 1
    comparison_data_1 = log_transform_if_needed(config, ra_data)
    do_comparison(config, data=comparison_data_1, extra_descriptor='_benchmark_step3_')

    # STEP 4: COPY AND DO TEMPORAL FILTERING, COMPARISON METHODS 2
    comparison_data_2 = temporal_filter_if_needed( 
        config,
        log_transform_if_needed(config, ra_data)    
    )
    do_comparison(config, data=comparison_data_2, extra_descriptor='_benchmark_step4_')

    # STEP 5: AGGREGATE AND DO COMPARISON METHODS 3
    data = preprocess_step2(config, ra_data)
    do_comparison(config, data=data, extra_descriptor='_benchmark_step5_')

    # STEP 6: RUN MITRE
    model = setup_model(config, data=data)

    if config.has_section('crossvalidation'):
        crossvalidate(config, model=model)
    elif config.has_section('leave_one_out'):
        leave_one_out(config, model=model)
    else:
        raise ValueError('Neither crossvalidation method was specified.')

    # debug output
    with open('benchmark_debug.pickle','w') as f:
        pickle.dump([ra_data, comparison_data_1, comparison_data_2, data],f)

def log_transform_if_needed(config, target_data):
    if config.has_option('preprocessing', 'log_transform'):
        if config.getboolean('preprocessing','log_transform'):
            logger.info('Applying log transform...')
            target_data = transforms.log_transform(target_data)
    return target_data

def temporal_filter_if_needed(config, target_data):
    if config.has_option('preprocessing','temporal_abundance_threshold'):
        logger.info('Appying temporal filtering')
        target_data, _ = filtering.discard_low_abundance(
            target_data,
            min_abundance_threshold = config.getfloat(
                'preprocessing',
                'temporal_abundance_threshold'),
            min_consecutive_samples = config.getfloat(
                'preprocessing',
                'temporal_abundance_consecutive_samples'),
            min_n_subjects = config.getfloat(
                'preprocessing',
                'temporal_abundance_n_subjects')
        )
    return target_data


def describe_dataset(dataset, comment=None):
    """ Log size of a dataset object. 

    A utility function, used often in the preprocessing step.

    """
    if comment is not None:
        logger.info(comment)
    logger.info('%d variables, %d subjects, %d total samples' % 
          (dataset.n_variables, dataset.n_subjects,
           sum(map(len, dataset.T))))

def preprocess(config):
    """ Load data, apply filters, create Dataset object.
    
    This is broken into two parts: a first step which 
    loads the data, applies initial filters, and converts to 
    relative abundance; then a second step which performs 
    phylogenetic aggregation and final filtering.

    """
    data = preprocess_step1(config)
    data = preprocess_step2(config, data)
    return data

def load_example(config, example_name):
    """ Populate configuration object with the settings for an example dataset.

    Valid options are 'bokulich', 'david', 'karelia', and 'digiulio'.

    """
    # Find the file with the configuration settings.
    default_cfg_file = os.path.join(os.path.dirname(__file__),
                                    'example_data',
                                    'example_settings.cfg')

    default_parser = ConfigParser.ConfigParser()
    default_parser.read(default_cfg_file)

    # Read the appropriate section and update config.
    examples = {'bokulich', 'david', 'karelia', 'digiulio'}
    if example_name not in examples:
        raise ValueError('Valid example datasets are: ' + 
                         ', '.join(examples))
    
    options = default_parser.options(example_name)
    if not config.has_section('description'):
        config.add_section('description')
    if not config.has_section('data'):
        config.add_section('data')
    
    config.set('description','tag', default_parser.get(example_name, 'tag'))
    options.remove('tag')

    # Find the directory with the relevant data files.
    data_directory = os.path.join(os.path.dirname(__file__),
                                  'example_data',
                                  example_name)
    for option in options:
        value = default_parser.get(example_name, option)
        if '$DIR/' in value:
            value = value.replace('$DIR/','')
            value = os.path.join(data_directory, value)
        config.set('data',option,value)

def preprocess_step1(config):
    """ Load data, apply initial filters and convert to RA, create Dataset object.

    """
    # 0. If necessary, update the configuration to contain the appropriate 
    # settings for one of the example data sets.

    if config.has_option('data', 'load_example'):
        load_example(config, config.get('data','load_example'))

    # 1. Input files. 
    counts_file = config.get('data','abundance_data')
    metadata_file = config.get('data','sample_metadata')
    subject_file = config.get('data', 'subject_data') 
    if config.has_option('data','sequence_key'):
        sequence_file = config.get('data','sequence_key')
    else:
        sequence_file = None

    # 2. Outcome
    outcome_variable = config.get('data', 'outcome_variable') 
    outcome_positive_value = config.get('data', 'outcome_positive_value') 
    # We don't know whether to expect the positive outcome value to be
    # a string, boolean, or integer, but the data type needs to match
    # the type in the dataframe of per-subject data, at least enough to 
    # allow meaningful equality testing. Somewhat clumsily, we 
    # just try to cast the string we read from the file to an int;
    # if the true value is Boolean, specify either 1 or 0 in the 
    # configuration file (not, e.g., 'true' or 'false').

    try:
        outcome_positive_value = int(outcome_positive_value)
    except ValueError:
        pass

    # 2a. Additional covariates. Assume that these are provided as
    # comma-separated lists. 

    # First, the categorical covariates. For categorical data, try to
    # convert strings to ints if possible (this should roughly match
    # the behavior of the import of the subject data file.)
    if config.has_option('data','additional_subject_covariates'):
        additional_subject_covariates = config.get('data','additional_subject_covariates')
        additional_subject_covariates = additional_subject_covariates.split(',')
        raw_default_states = config.get('data','additional_covariate_default_states')
        raw_default_states = raw_default_states.split(',')
        additional_covariate_default_states = []
        for state in raw_default_states:
            try: 
                state = int(state)
            except ValueError:
                pass
            additional_covariate_default_states.append(state)
    else:
        additional_covariate_default_states = []
        additional_subject_covariates = []

    # Second, the continuous covariates.  
    if config.has_option('data','additional_subject_continuous_covariates'):
        additional_subject_continuous_covariates = config.get(
            'data',
            'additional_subject_continuous_covariates'
        )
        additional_subject_continuous_covariates = additional_subject_continuous_covariates.split(',')
    else:
        additional_subject_continuous_covariates = []

    data = basic.load_dada2_result(
            counts_file,
            metadata_file,
            subject_file,
            sequence_id_filename=sequence_file,
            outcome_variable=outcome_variable,
            outcome_positive_value=outcome_positive_value, 
            additional_subject_categorical_covariates = additional_subject_covariates,
            additional_covariate_default_states = additional_covariate_default_states,
            additional_subject_continuous_covariates = additional_subject_continuous_covariates,
            )
    describe_dataset(data, 'Data imported (before any filtering:)')

    
    # 3. Filtering

    # 3a. Overall abundance filter
    if config.has_option('preprocessing','min_overall_abundance'):
        # Drop sequences/OTUs with fewer reads (summing across all
        # samples) than the threshold
        minimum_reads_per_sequence = config.getfloat(
            'preprocessing','min_overall_abundance'
        )
        data, _ = filtering.discard_low_overall_abundance(
            data,
            minimum_reads_per_sequence
        )
        describe_dataset(
            data,
            'After filtering RSVs/OTUs with too few counts:'
        )

    # 3b. Sample depth filter
    if config.has_option('preprocessing','min_sample_reads'):
        # Drop all samples where the total number of reads was below a
        # threshold
        minimum_reads_per_sample = config.getfloat(
            'preprocessing',
            'min_sample_reads'
        )
        data = filtering.discard_low_depth_samples(
            data,
            minimum_reads_per_sample
        )
        describe_dataset(
            data,
            'After filtering samples with too few counts:'
        )

    # 3c. Trimming the experimental window
    if config.has_option('preprocessing','trim_start'):
        experiment_start = config.getfloat('preprocessing','trim_start')
        experiment_end = config.getfloat('preprocessing','trim_stop')
        data = filtering.trim(data, experiment_start, experiment_end)
        describe_dataset(
            data,
            'After trimming dataset to specified experimental time window:'
        )

    # 3d. Drop subjects with inadequately dense temporal sampling
    if config.has_option('preprocessing','density_filter_n_samples'):
        subject_min_observations_per_long_window = (
            config.getfloat('preprocessing',
                            'density_filter_n_samples')
        )
        n_intervals = config.getint(
            'preprocessing',
            'density_filter_n_intervals')
        n_consecutive = config.getint(
            'preprocessing',
            'density_filter_n_consecutive'
        )
        data = filtering.filter_on_sample_density(
            data,
            subject_min_observations_per_long_window,
            n_intervals,
            n_consecutive=n_consecutive
        )
        describe_dataset(
            data,
            ('After filtering subjects with ' + 
            'inadequately dense temporal sampling:')
        )

    # Optionally subsample the data, keeping a set number of subjects
    # chosen at random.
    if config.has_option('preprocessing','subsample_subjects'):
        n_subjects_to_keep = config.getint('preprocessing','subsample_subjects')
        indices_to_keep = np.random.choice(data.n_subjects, n_subjects_to_keep, replace=False)
        data = transforms.select_subjects(data, indices_to_keep)
        logger.info('Subsampling, kept indices: %s' % str(indices_to_keep))

    # 3e. Relative abundance transformation.
    if config.has_option('preprocessing', 'take_relative_abundance'):
        if config.getboolean('preprocessing','take_relative_abundance'):
            data = transforms.take_relative_abundance(data) 
            logger.info('Transformed to relative abundance.')

    return data


def preprocess_step2(config, data):
    """ Aggregate, optionally transform, temporal abundance filter, etc.

    Taxonomy information is applied here (after the tree 
    is established; currently no facility for annotating
    taxonomies without a tree; to be added.)
    

    """
    # 3f. Phylogenetic aggregation.
    has_tree = False
    if config.has_option('preprocessing', 'aggregate_on_phylogeny'):
        if config.getboolean('preprocessing','aggregate_on_phylogeny'):
            logger.info('Phylogenetic aggregation begins.')
            jplace_file = config.get('data', 'jplace_file') 
            data, _, _ = pplacer.aggregate_by_pplacer_simplified(
                jplace_file,
                data
            )
            has_tree = True
            describe_dataset(data,'After phylogenetic aggregation:')
    

    # 3f(b). Optional taxonomy information.
    if has_tree and config.has_option('data','taxonomy_source'):
        # Valid options are 'pplacer' and 'table' and 'hybrid'
        taxonomy_source = config.get('data','taxonomy_source')
        logger.info('Parsing taxonomic annotations.')
    else:
        taxonomy_source = None

    if taxonomy_source == 'table':
        placement_table_filename = config.get('data','placement_table')
        if config.has_option('data','sequence_key'):
            sequence_fasta_filename = config.get('data','sequence_key')
        else:
            sequence_fasta_filename = None
        taxonomy_annotation.annotate_dataset_table(
            data,
            placement_table_filename,
            sequence_fasta_filename
        )

    elif taxonomy_source == 'pplacer':
        jplace_filename = config.get('data', 'jplace_file')
        taxa_table_filename = config.get('data', 'pplacer_taxa_table')
        seq_info_filename = config.get('data', 'pplacer_seq_info')
        taxonomy_annotation.annotate_dataset_pplacer(
            data,
            jplace_filename,
            taxa_table_filename,
            seq_info_filename
        )

    elif taxonomy_source == 'hybrid':
        jplace_filename = config.get('data', 'jplace_file')
        taxa_table_filename = config.get('data', 'pplacer_taxa_table')
        seq_info_filename = config.get('data', 'pplacer_seq_info')
        placement_table_filename = config.get('data','placement_table')
        if config.has_option('data','sequence_key'):
            sequence_fasta_filename = config.get('data','sequence_key')
        else:
            sequence_fasta_filename = None
        taxonomy_annotation.annotate_dataset_hybrid(
            data,
            jplace_filename,
            taxa_table_filename,
            seq_info_filename,
            placement_table_filename,
            sequence_fasta_filename
        )
        
    # 3g. Log transform
    data = log_transform_if_needed(config, data)

    # 3h. Temporal abundance filter.
    # We drop all variables except those which exceed a threshold
    # abundance for a certain number of consecutive observations in a
    # certain number of subjects
    data = temporal_filter_if_needed(config, data)

    # 3i. Surplus internal node removal.
    if config.has_option('preprocessing', 'discard_surplus_internal_nodes'):
        if config.getboolean('preprocessing',
                             'discard_surplus_internal_nodes'):
            logger.info('Removing surplus internal nodes...')
            data, _ = filtering.discard_surplus_internal_nodes(data)
            describe_dataset(
                data,
                ('After removing internal nodes ' +
                 'not needed to maintain topology:')
            ) 

    # Debugging feature: randomize the labels
    if (config.has_option('preprocessing', 'randomize_labels') and
        config.getboolean('preprocessing', 'randomize_labels')):
        np.random.shuffle(data.y)

    # 3h. Pickling.
    prefix = config.get('description','tag')
    if config.has_option('preprocessing', 'pickle_dataset'):
        if config.getboolean('preprocessing','pickle_dataset'):
            logger.info('Saving dataset...')
            filename = prefix + '_dataset_object.pickle'
            with open(filename, 'w') as f:
                pickle.dump(data,f)
            logger.info('Dataset written to %s' % filename)

    # 3i. Write taxonomic annotations (if they exist), now
    # that all filtering has been done.
    if taxonomy_source is not None:
        prefix = config.get('description','tag')
        filename = prefix + '_variable_annotations.txt'
        write_variable_table(data,filename)
    return data

def write_variable_table(dataset, filename):
    """ Dump notes on every variable to a text file.

    Produces a tab-delimited table, first column the variable name,
    second its annotation in dataset.variable_annotations (if any),
    third column the names of its descendant leaves (if any).

    """ 
    fields = ['description','leaves']
    df = pd.DataFrame(columns=fields, index=dataset.variable_names)
    for name in dataset.variable_names:
        df.loc[name,'description'] = (
            dataset.variable_annotations.get(name, '(no annotation)')
        )
        node_list = dataset.variable_tree.search_nodes(name=name)
        if not node_list:
            leaves_as_string = '(not in variable tree)'
        elif len(node_list) > 1:
            raise ValueError('Ambiguous map from variables to tree')
        else:
            node = node_list[0]
            if node.is_leaf():
                # Leave this field as empty/NA
                continue
            leaves_as_string = ' '.join(node.get_leaf_names())
        df.loc[name,'leaves'] = leaves_as_string
    df.to_csv(filename,sep='\t',index_label='name')
    return df

def do_comparison(config, data=None, extra_descriptor=''):
    """ Run comparison methods with k-fold or LOO CV. 

    Calls either k_fold_comparison (the default) or
    leave_one_out_comparison (if the option cv_type is
    'leave_one_out'.)
    
    If 'extra_descriptor' is given it will be inserted in the 
    output filename.
    
    """
    if (config.has_option('comparison_methods','cv_type')
        and config.get('comparison_methods','cv_type') == 
        'leave_one_out'):
        leave_one_out_comparison(config, data, extra_descriptor)
    else:
        k_fold_comparison(config, data, extra_descriptor)

def k_fold_comparison(config, data=None, extra_descriptor=''):
    """ Apply comparison methods to dataset object.

    Tries to fit L1-regularized logistic regression and RF classifiers
    to the dataset, and reports back on the accuracy of the results. 

    The first set of results involve no cross-validation [well, except
    internally, to determine the regularization parameter]: they
    assesses the accuracy of the classifiers on the training data,
    as a sort of best-case scenario.

    The second set of results report median AUC across folds of cross
    validation and a summary confusion matrix (summing the confusion
    matrices resulting from the application of each fold's classifier
    to its appropriate test set.)

    If there is a configuration option 'load_data_from_pickle' in
    section 'comparison_methods' the function tries to load that data,
    ignoring the data argument (the option value shold be the path to
    a pickle file containing a single Dataset object.) If the data
    argument is None, and that configuration option is not present, an
    exception results.

    If 'extra_descriptor' is given it will be inserted in the 
    output filename.

    """
    if config.has_option('comparison_methods','load_data_from_pickle'):
        with open(config.get('comparison_methods',
                             'load_data_from_pickle')) as f:
            data = pickle.load(f)

    if data is None:
        raise ValueError(
            'Dataset must be passed as argument if not specified '
            'in config file.'
        )

    comparison_n_intervals = config.getint('comparison_methods',
                                           'n_intervals')
    comparison_n_consecutive = config.getint('comparison_methods',
                                             'n_consecutive')
    comparison_n_folds = config.getint('comparison_methods',
                                       'n_folds')
    
    report = [] # list by line or block

    # Round 1: naive assessment of accuracy on training data.
    l1 = comparison_methods.L1LogisticRegression(
        data,
        comparison_n_intervals, 
        comparison_n_consecutive
    )
    rf = comparison_methods.RandomForest(
        data,
        comparison_n_intervals, 
        comparison_n_consecutive
    )

    l1_y = l1.predict(data)
    rf_y = l1.predict(data)

    report += ['Results on training sets:',
               'L1-regularized logistic regression:']
    report.append(classifier_accuracy_report(data.y, l1_y))
    report.append('Random forests:')
    report.append(classifier_accuracy_report(data.y, rf_y))

    # Round 2: cross-validation
    report.append('Cross-validation results (%d folds):' %
                  comparison_n_folds)

    for label, method in comparison_methods_and_labels:
        cv_aucs = []
        cv_confusion_matrices = []
        enumerated_folds = enumerate(
            stratified_k_folds(
                data,
                folds=comparison_n_folds
            )
        )
        results = []
        for fold, (train, test) in enumerated_folds:
            classifier=method(train,
                              comparison_n_intervals, 
                              comparison_n_consecutive)
            # For both these classifiers, what is returned is 
            # an n_subjects x 2 array, the first column giving the
            # predicted probabilities of label 0, the second of label 1
            # (these sum to 1.) We want simply the probabilities of 
            # label 1.
            probabilities = classifier.classifier.predict_proba(
                classifier.transform_X(test)
            )[:,1]
            results.append((fold, test.y, probabilities))
        this_method_report = tabulate_metrics(results, label)
        report.append(this_method_report)
        report.append('\n')

    prefix = config.get('description','tag')
    filename = prefix + extra_descriptor +'comparison.txt'
    with open(filename, 'w') as f:
        f.write('\n'.join(report))
        f.write('\n')
    logger.info('Comparison method results written to %s' % filename)

def classifier_accuracy_report(true_y, prediction):
    auc = roc_auc_score(true_y.astype(float), prediction.astype(float))
    conf = confusion_matrix(true_y, prediction)
    lines = ['AUC: %.3f' % auc,
             'Confusion matrix: \n\t%s' % str(conf).replace('\n','\n\t')]
    return '\n'.join(lines) + '\n'

def sample(config, model=None):
    """ Create sampler and sample per options in configuration file.

    If there is a configuration option 'load_model_from_pickle' in
    section 'sampling' the function tries to load that model, ignoring
    the data argument (the option value shold be the path to a pickle
    file containing a single LogisticRuleModel object.) If the model
    argument is None, and that configuration option is not present, an
    exception results.

    """
    if config.has_option('sampling','load_model_from_pickle'):
        with open(config.get('sampling','load_model_from_pickle')) as f:
            model = pickle.load(f)

    if model is None:
        raise ValueError('Model must be passed as argument if not specified in config file.')
    
    l = [hamming(model.data.y,t) for t in model.rule_population.flat_truth]
    arbitrary_rl = rules.RuleList(
        [[model.rule_population.flat_rules[np.argmin(l)]]]
     )
    sampler = logit_rules.LogisticRuleSampler(model,
                                              arbitrary_rl)

    if config.has_option('sampling','sampling_time'):
        sampling_time = config.getfloat('sampling','sampling_time')
        logger.info('Starting sampling: will continue for %.1f seconds' %
                    sampling_time)
        sampler.sample_for(sampling_time)
    elif config.has_option('sampling','total_samples'):
        total_samples = config.getint('sampling','total_samples')
        logger.info('Starting to draw %d samples' % total_samples)
        sampler.sample(total_samples)
    else:
        raise ValueError('Either number of samples or sampling time must be specified.')


    if config.has_option('sampling', 'pickle_sampler'):
        prefix = config.get('description','tag')
        if config.getboolean('sampling','pickle_sampler'):
            filename = prefix + '_sampler_object.pickle'
            with open(filename, 'w') as f:
                pickle.dump(sampler,f)
            logger.info('Sampler written to %s' % filename)

    return sampler


def crossvalidate(config, model=None):
    """ Cross-validate MITRE model per options in configuration file.

    Most options and arguments as for the 'sample' method and the 
    'sampling' configuration section, with the exception of 
    'n_folds', specifying the number of folds of cross-validation,
    'pickle_folds' replacing 'pickle_sampler', 'parallel_workers'
    specifying the number of folds to attempt to run simultaneously,
    and 'burnin_faction', to be used for summarizing the 
    results of each training run.

    Note that if the pickle_folds option is set, both individual
    files storing the sampler objects for each fold and a file
    collecting the training and test datasets for each fold will
    be written. If pickle_results is set, files with the 
    CV results- outcome probabilities for each fold's test data,
    from point and ensemble summaries respectively- are written
    (in a tuple, each entry (fold_index, fold_test_y_values,
    outcome_probability_vector)).

    Detailed postprocessing is not done here; instead, AUC and
    confusion matrices for point and ensemble estimates from each fold
    of cross-validation applied to its proper test set, and summaries
    of those values, are written to a file.

    """
    ###
    # Extract options from configuration file
    if config.has_option('crossvalidation','load_model_from_pickle'):
        with open(config.get('crossvalidation',
                             'load_model_from_pickle')) as f:
            model = pickle.load(f)

    if model is None:
        raise ValueError('Model must be passed as argument '
                         'if not specified in config file.')
    
    if config.has_option('crossvalidation','sampling_time'):
        stop_criterion = 'time'
        stop_samples = None
        stop_time = config.getfloat('crossvalidation',
                                               'sampling_time')
    elif config.has_option('crossvalidation','total_samples'):
        stop_criterion = 'samples'
        stop_time = None
        stop_samples = config.getint('crossvalidation','total_samples')
    else:
        raise ValueError('A stopping criterion must be specified.')

    cv_n_folds = config.getint('crossvalidation','n_folds')
    cv_n_workers = config.getint('crossvalidation','parallel_workers')
    cv_burnin = config.getfloat('crossvalidation','burnin_fraction')

    prefix = config.get('description','tag')
    if (config.has_option('crossvalidation', 'pickle_folds') and
        config.getboolean('crossvalidation','pickle_folds')):
        pickle_folds = True
    else: 
        pickle_folds = False

    pickle_results = (
        config.has_option('crossvalidation', 'pickle_results') and
        config.getboolean('crossvalidation','pickle_results')
    )

    if (config.has_option('crossvalidation', 'write_reports_every_fold') and
        config.getboolean('crossvalidation','write_reports_every_fold')):
        write_reports = True
    else: 
        write_reports = False

    if (config.has_option('crossvalidation', 'write_full_summaries_every_fold') and
        config.getboolean('crossvalidation', 'write_full_summaries_every_fold')):
        write_full_summary = True
    else: 
        write_full_summary = False

    ###
    # DEBUGGING CODE:
    # This supports a fairly paranoid test of whether information may be leaking from
    # the model built on the full training set into the calculations for the 
    # separate folds
    if (config.has_option('crossvalidation', 'randomize_cv') and
        config.getboolean('crossvalidation','randomize_cv')):
        base_data = model.data.copy()
        logger.critical('Executing paranoid reshuffle of CV data')
        np.random.shuffle(base_data.y)
    else:
        base_data = model.data
            
    ### 
    # Generate test and training sets
    folds = stratified_k_folds(
        base_data,
        folds=cv_n_folds
    )

    if (config.has_option('crossvalidation', 'randomize_cv') and
        config.getboolean('crossvalidation','randomize_cv')):
        for train, test in folds:
            np.random.shuffle(test.y)

    if pickle_folds:
        filename = prefix + '_folds.pickle'
        with open(filename,'w') as f:
            pickle.dump(folds, f)
    
    ###
    # Set up parallelized loop over folds
    job_queue = mp.Queue()
    result_queue = mp.Queue()
    worker_args = (model, cv_burnin, stop_criterion, stop_time,
                   stop_samples, job_queue, result_queue, pickle_folds,
                   prefix, write_reports, write_full_summary)

    processes = [
        mp.Process(target=_crossvalidation_worker, args=worker_args) for 
        worker_index in xrange(cv_n_workers)
    ]
    for fold, (train, test) in enumerate(folds):
        job_queue.put((fold, train, test))
    for _ in xrange(cv_n_workers):
        job_queue.put('end')
    for p in processes:
        p.start()
    results = []
    for _ in xrange(cv_n_folds):
        results.append(result_queue.get())
    n_processes_joined = 0
    for p in processes:
        n_processes_joined += 1
        p.join()

    results.sort(key = lambda t: t[0])
    (fold_indices,
     fold_test_y_values,
     fold_point_probabilities,
     fold_ensemble_probabilities) = zip(*results)

    point_results = zip(fold_indices, fold_test_y_values,
                        fold_point_probabilities)
    ensemble_results = zip(fold_indices, fold_test_y_values,
                        fold_ensemble_probabilities)

    if pickle_results:
        with open(prefix + '_cv_point_results.pickle','w') as f:
            pickle.dump(point_results,f)
        with open(prefix + '_cv_ensemble_results.pickle','w') as f:
            pickle.dump(ensemble_results,f)
 
    ###
    # Write text output
    report = (
        tabulate_metrics(point_results, 'Point summary') +
        '\n\n' +
        tabulate_metrics(ensemble_results, 'Ensemble summary') 
    )
    
    filename = prefix + '_mitre_cv_report.txt'
    with open(filename, 'w') as f:
        f.write(report)

    return (folds, point_results, ensemble_results)

####
def _crossvalidation_worker(model, burnin_fraction, stop_criterion,
                            stop_time, stop_samples, argument_queue,
                            result_queue, pickle_samplers, 
                            pickle_tag, write_report, write_full_summary):
    """ Worker function for parallelizing MITRE crossvalidation.

    Given a model, a number of seconds to sample or number of samples
    to draw, and a burnin fraction, the worker repeatedly pulls
    train and test datasets from a queue, updates the model with the 
    training data, samples from the posterior, tests the performance
    of the resulting point summary and ensemble classifiers on the 
    test data, and writes a summary back to the result queue. If 
    the pickle options are set, each sampler object is written to a file.
    If the write_report option is set, the quick report
    from the posterior summary object is written to a file for each fold.

    Arguments:
    model - LogisticRuleModel 
    burnin_fraction - used in extracting the point summary and ensemble
    classifiers
    stop_criterion - 'time' or 'samples'
    stop_time - seconds to sample each fold (ignored if stop_criterion
    is 'samples')
    stop_samples - iterations to sample for each fold (ignored if
    stop_criterion is 'time')
    argument_queue - each item should be a tuple (fold_index,
    training_data, test_data) or the string 'end', which triggers
    this function to return 
    result_queue - each entry will be a list of the form:
        [point_auc, point_confusion_matrix, ensemble_auc,
         ensemble_confusion_matrix]

    """
    print 'worker'
    while True:
        item = argument_queue.get()
        if item == 'end':
            return
        fold_index, train, test = item
        print 'processing fold %d' % fold_index

        # Keep the same population of allowed primitive
        # rules, but update the corresponding truth values for
        # the training data in this fold.
        model.data = train
        model.rule_population.data = train
        model.rule_population._update_truths()
       
        l = [hamming(model.data.y,t) for t in 
             model.rule_population.flat_truth]
        arbitrary_rl = rules.RuleList(
            [[model.rule_population.flat_rules[np.argmin(l)]]]
        )
        sampler = logit_rules.LogisticRuleSampler(
            model,
            arbitrary_rl
        )
        print 'sampling starts'
        if stop_criterion == 'time':
            sampler.sample_for(stop_time)
        else:
            sampler.sample(stop_samples)
        print 'sampling stops'

        # how to evaluate AUC/conf on this data? 
        summary = posterior.PosteriorSummary(
            sampler,
            burnin_fraction,
            tag=pickle_tag + '_%d_' % fold_index,
        )
        summary.point_summarize()
        point_probabilities = summary.point_probabilities(
            test_data=test
        )
        ensemble_probabilities = summary.ensemble_probabilities(
            test_data=test
        )
        result_queue.put([fold_index, test.y, point_probabilities, 
                          ensemble_probabilities])
        if write_report:
            with open(pickle_tag + '_%d_' % fold_index + '_qr.txt','w') as f:
                f.write(summary._quick_report())
        if write_full_summary:
            with open(pickle_tag + '_%d_' % fold_index + '_full_summary.txt','w') as f:
                f.write(summary.all_summaries())
        if pickle_samplers:
            filename = (pickle_tag + ('_fold_%d_' % fold_index) +
                        'sampler_object.pickle')
            with open(filename, 'w') as f:
                pickle.dump(sampler,f)
            
####

def leave_one_out(config, model=None):
    """ Leave-one-out cross-validation, per options in configuration file.

    Most options and arguments as for the 'sample' method and the
    'sampling' configuration section, 'pickle_folds' replacing
    'pickle_sampler', 'parallel_workers' specifying the number of
    folds to attempt to run simultaneously, and 'burnin_faction', to
    be used for summarizing the results of each training run.

    Note that if the pickle_folds option is set, both individual
    files storing the sampler objects for each fold and a file
    collecting the training and test datasets for each fold will
    be written. Be aware this is easily many gigabytes of data.
    If pickle_results is set, the point summaries for each fold's
    training set are saved.

    Detailed postprocessing is not done here; instead, outcome probabilities
    for the held out point in each fold, from both point and ensemble estimates, and
    a summary confusion matrix are written to a file.

    """
    ###
    # Extract options from configuration file
    if config.has_option('leave_one_out','load_model_from_pickle'):
        with open(config.get('leave_one_out',
                             'load_model_from_pickle')) as f:
            model = pickle.load(f)
            logger.info('Finished loading model')

    if model is None:
        raise ValueError('Model must be passed as argument '
                         'if not specified in config file.')
    
    if config.has_option('leave_one_out','sampling_time'):
        stop_criterion = 'time'
        stop_samples = None
        stop_time = config.getfloat('leave_one_out',
                                               'sampling_time')
    elif config.has_option('leave_one_out','total_samples'):
        stop_criterion = 'samples'
        stop_time = None
        stop_samples = config.getint('leave_one_out','total_samples')
    else:
        raise ValueError('A stopping criterion must be specified.')

    cv_n_folds = model.data.n_subjects
    cv_n_workers = config.getint('leave_one_out','parallel_workers')
    cv_burnin = config.getfloat('leave_one_out','burnin_fraction')

    prefix = config.get('description','tag')
    if (config.has_option('leave_one_out', 'pickle_folds') and
        config.getboolean('leave_one_out','pickle_folds')):
        pickle_folds = True
    else: 
        pickle_folds = False

    pickle_results = (
        config.has_option('leave_one_out', 'pickle_results') and
        config.getboolean('leave_one_out','pickle_results')
    )

    if (config.has_option('leave_one_out', 'write_reports_every_fold') and
        config.getboolean('leave_one_out','write_reports_every_fold')):
        write_reports = True
    else: 
        write_reports = False

    if (config.has_option('leave_one_out', 'write_full_summaries_every_fold') and
        config.getboolean('leave_one_out', 'write_full_summaries_every_fold')):
        write_full_summary = True
    else: 
        write_full_summary = False

    logger.info('Generating folds...')
    ### 
    # Generate test and training sets
    folds = leave_one_out_folds(
        model.data
    )

    if pickle_folds:
        filename = prefix + '_folds.pickle'
        with open(filename,'w') as f:
            pickle.dump(folds, f)
    
    logger.info('Generated folds.')
    ###
    # Set up parallelized loop over folds
    job_queue = mp.Queue()
    result_queue = mp.Queue()
    worker_args = (model, cv_burnin, stop_criterion, stop_time,
                   stop_samples, job_queue, result_queue, pickle_folds,
                   prefix, write_reports, write_full_summary)

    processes = [
        mp.Process(target=_crossvalidation_worker, args=worker_args) for 
        worker_index in xrange(cv_n_workers)
    ]
    for fold, (train, test) in enumerate(folds):
        job_queue.put((fold, train, test))
    for _ in xrange(cv_n_workers):
        job_queue.put('end')
    for p in processes:
        p.start()
    results = []
    for _ in xrange(cv_n_folds):
        results.append(result_queue.get())
    n_processes_joined = 0
    for p in processes:
        n_processes_joined += 1
        p.join()

    print 'finish joining processes'
    results.sort(key = lambda t: t[0])
    (fold_indices,
     fold_test_y_values,
     fold_point_probabilities,
     fold_ensemble_probabilities) = zip(*results)

    test_true_y = np.array([ys[0] for ys in fold_test_y_values])
    ensemble_probabilities = np.array([p[0] for p in fold_ensemble_probabilities])
    point_probabilities = np.array([p[0] for p in fold_point_probabilities])
    combined_results = [('ensemble', test_true_y, ensemble_probabilities),
                        ('point', test_true_y, point_probabilities)]

    report = leave_one_out_report(combined_results)
    prefix = config.get('description','tag')
    filename = prefix + '_mitre_leave_one_out_report.txt'
    with open(filename, 'w') as f:
        f.write(report)

    return (folds, combined_results)

def leave_one_out_report(combined_results):
    """ Evaluate leave-one-out CV results from different methods.

    Arguments: 
    combined_results: list of tuples of the form
    (method_name, true_y_vector, predicted_probabilities_vector)

    Note the vectors really do need to be numpy arrays.

    Returns: formatted report as string

    """
    ### 
    # Unfortunate code duplication with tabulate_metrics here,
    # to be resolved later
    probability_metrics = [
        ('AUC', roc_auc_score),
        ('AP', metrics.average_precision_score)
    ]
    binary_metrics = [
        ('F1', metrics.f1_score),
        ('MCC', metrics.matthews_corrcoef),
        ('precision', metrics.precision_score),
        ('recall', metrics.recall_score)
    ] 
    metric_results = {label: [] for label, _ in
               probability_metrics + binary_metrics}
    metric_results.update({'tn': [], 'fp': [], 'fn': [], 'tp': []})
    for label, metric in probability_metrics:
        for fold, y_true, y_pred in combined_results:
            metric_results[label].append(metric(y_true, y_pred))
    for method, y_true, probabilities in combined_results:
        y_pred = probabilities > 0.5
        for label, metric in binary_metrics:
            metric_results[label].append(metric(y_true, y_pred))
        conf = zip(
            ('tn', 'fp', 'fn', 'tp'),
            metrics.confusion_matrix(y_true, y_pred).flat
        )
        for label, n in conf:
            metric_results[label].append(n)
    index=[t[0] for t in combined_results]
    table = pd.DataFrame(data=metric_results, 
                         index=index)
    report = table.to_string(float_format=lambda x: '%.3g' % x)
    return report

def leave_one_out_comparison(config, data=None, extra_descriptor=''):
    """ Apply comparison methods to dataset object with LOO CV.

    Tries to fit L1-regularized logistic regression and RF classifiers
    to the dataset, and reports back on the accuracy of the results. 

    This is separate from do_comparison because the format of 
    results used for k-fold CV there does not make a lot of sense
    for leave-one-out CV. 

    If there is a configuration option 'load_data_from_pickle' in
    section 'comparison_methods' the function tries to load that data,
    ignoring the data argument (the option value shold be the path to
    a pickle file containing a single Dataset object.) If the data
    argument is None, and that configuration option is not present, an
    exception results.

    If 'extra_descriptor' is given it will be inserted in the 
    output filename.

    """
    if config.has_option('comparison_methods','load_data_from_pickle'):
        with open(config.get('comparison_methods',
                             'load_data_from_pickle')) as f:
            data = pickle.load(f)

    if data is None:
        raise ValueError(
            'Dataset must be passed as argument if not specified '
            'in config file.'
        )

    comparison_n_intervals = config.getint('comparison_methods',
                                           'n_intervals')
    comparison_n_consecutive = config.getint('comparison_methods',
                                             'n_consecutive')

    logger.info('Generating folds....')
    folds = leave_one_out_folds(data)
    results = []
    for label, method in comparison_methods_and_labels:
        test_true_y = []
        test_probabilities = []
        for fold, (train, test) in enumerate(folds):
            classifier=method(train,
                              comparison_n_intervals, 
                              comparison_n_consecutive)
            # For both these classifiers, what is returned is 
            # an n_subjects x 2 array, the first column giving the
            # predicted probabilities of label 0, the second of label 1
            # (these sum to 1.) We want simply the probabilities of 
            # label 1.
            probabilities = classifier.classifier.predict_proba(
                classifier.transform_X(test)
            )[:,1]
            test_true_y.append(test.y[0])
            test_probabilities.append(probabilities[0])
        results.append((label, np.array(test_true_y), np.array(test_probabilities)))

    logger.info('Writing report...')
    report = leave_one_out_report(results)
    prefix = config.get('description','tag')
    filename = prefix + extra_descriptor + '_leave_one_out_comparison.txt'
    with open(filename, 'w') as f:
        f.write(report)
    logger.info('Comparison method results written to %s' % filename)

###

def setup_model(config, data=None):
    """ Create model applying to specified dataset with parameters from config file.

    If there is a configuration option 'load_data_from_pickle' in
    section 'model' the function tries to load that data, ignoring the
    data argument (the option value shold be the path to a pickle file
    containing a single Dataset object.) If the data argument is None,
    and that configuration option is not present, an exception
    results.

    """
    if config.has_option('model','load_data_from_pickle'):
        with open(config.get('model','load_data_from_pickle')) as f:
            data = pickle.load(f)

    if data is None:
        raise ValueError('Dataset must be passed as argument if not specified in config file.')

    n_intervals = config.getint('model','n_intervals')
    t_min = config.getfloat('model','t_min')
    # Unfortunately the maximum time window length is 
    # referred to as 'tmax' in the code and 't_max' in the 
    # configuration file. (Same problem for 'tmin'/'t_min' but 
    # for historical reasons that's a positional argument, 
    # so it causes fewer problems.)
    tmax = config.getfloat('model','t_max')
    
    additional_model_kwargs = {}
    additional_options = [
        'prior_coefficient_variance',
        'hyperparameter_alpha_primitives',
        'hyperparameter_beta_primitives',
        'hyperparameter_alpha_m',
        'hyperparameter_beta_m',
        'window_concentration_typical',
        'window_concentration_update_ratio',
        'hyperparameter_a_empty',
        'hyperparameter_b_empty',
        'max_thresholds',
        'max_rules',
        'max_primitives',
        'delta_l_scale_mean',
        'delta_l_scale_sigma',
        'lambda_l_offset',
    ]

    for option in additional_options:
        if config.has_option('model',option):
            additional_model_kwargs[option] = (
                config.getfloat('model',option)
            )

    model = logit_rules.LogisticRuleModel(
        data, 
        t_min,
        N_intervals=n_intervals,
        n_workers=1,
        tmax=tmax,
        **additional_model_kwargs)
    logger.info('Created model object.')

    if config.has_option('model','filter_thresholds_min_difference_average'):
        raise ValueError('Deprecated filtering option') 
    if config.has_option('model','filter_on_tree_min_differences'):
        raise ValueError('Deprecated filtering option')

    if config.has_option('model', 'pickle_model'):
        if config.getboolean('model','pickle_model'):
            prefix = config.get('description','tag')
            filename = prefix + '_model_object.pickle'
            with open(filename, 'w') as f:
                pickle.dump(model,f)
            logger.info('Model written to %s' % filename)
    return model

def do_postprocessing(config, sampler=None):
    """ Summarize sampler results per options in configuration file.

    If there is a configuration option 'load_sampler_from_pickle' in
    section 'postprocessing' the function tries to load that sampler,
    ignoring the sampler argument (the option value shold be the path
    to a pickle file containing a single LogisticRuleSampler object.)
    If the sampler argument is None, and that configuration option is
    not present, an exception results.

    Note that if no burnin fraction is specified, 5% is used.

    """
    if config.has_option('postprocessing','load_sampler_from_pickle'):
        with open(config.get('postprocessing','load_sampler_from_pickle')) as f:
            sampler = pickle.load(f)

    if sampler is None:
        raise ValueError('Sampler must be passed as argument if not specified in config file.')
  
    if config.has_option('postprocessing','burnin_fraction'):
        burnin_fraction = config.getfloat('postprocessing','burnin_fraction')
    else:
        burnin_fraction = 0.05

    summarizer = posterior.PosteriorSummary(sampler, burnin_fraction)
    prefix = config.get('description', 'tag')

    if (config.has_option('postprocessing','quick_summary') and
        config.getboolean('postprocessing','quick_summary')):
        report = summarizer._quick_report()
        with open(prefix + '_quick_summary.txt', 'w') as f:
            f.write(report)

    if (config.has_option('postprocessing','full_summary') and
        config.getboolean('postprocessing','full_summary')):
        report = summarizer.all_summaries()
        with open(prefix + '_full_summary.txt', 'w') as f:
            f.write(report)

    if (config.has_option('postprocessing','mixing_diagnostics') and
        config.getboolean('postprocessing','mixing_diagnostics')):
        summarizer.mixing_diagnostics()

    bayes_kwargs = {}
    if (config.has_option('postprocessing','bayes_factor_samples')):
        bfs = config.getint('postprocessing','bayes_factor_samples')
        bayes_kwargs['N_primitive_prior_samples'] = bfs

    bayes_factors_done = False
    if (config.has_option('postprocessing','bayes_factor_table') and
        config.getboolean('postprocessing','bayes_factor_table')):
        report = summarizer.bayes_summary(**bayes_kwargs)
        bayes_factors_done = True
        with open(prefix + '_bayes_factor_table.txt', 'w') as f:
            f.write(report)

    if (config.has_option('postprocessing','gui_output') and
        config.getboolean('postprocessing','gui_output')):
        if not bayes_factors_done:
            report = summarizer.bayes_summary(**bayes_kwargs)
            bayes_factors_done = True
        write_d3_output(config, summarizer)

def write_d3_output(config, summarizer):
    ## First write the file using a template in the local directory; then worry
    # about installing the HTML template as a package asset later
    logger.info('Preparing interactive visualization')
    model = summarizer.model
    variables = model.data.variable_names
    tree = model.data.variable_tree
    prefix = config.get('description', 'tag')
    fname = prefix + '_visualization.html'
 
    ########################################
    # 1. Which variables - leaves in the tree- become rows in the heat
    # map?  Write them in a sensible order.
    leaf_ordering = tree.get_leaf_names()
    leaf_indices = [variables.index(n) for n in leaf_ordering]

    row_data_json = json.dumps(leaf_indices)

    ########################################
    # 2. Now write out data about _all_ the variables.
    fields = ['variable_index','parent','x','y',
              'name','distance','crossbar_length','annotation']
    df = pd.DataFrame(columns=fields, index=variables)
    # We compile a table of information about each node, including a
    # suggested geometry for displaying it.
    #
    # First, walk the tree in such a way that a node's children are
    # always visited before it,
    for node in tree.traverse(strategy='postorder'):
        variable = node.name
        annotation = model.data.variable_annotations.get(
            variable, '(no annotation)'
        )
        df.loc[variable,'annotation'] = annotation
        df.loc[variable,'name'] = variable.strip('"') # fix this later...
        df.loc[variable,'variable_index'] = variables.index(variable)
        # D3 code assumes that the root node has 
        # parent node index 'nan'
        if node.up:
            df.loc[variable,'parent'] = node.up.name
            df.loc[variable,'distance'] = node.dist
        else:
            df.loc[variable,'parent'] = 'nan'
            df.loc[variable,'distance'] = 0.
        if node.children:
            y_values = [df.loc[n.name,'y'] for n in node.children]
            max_y = np.max(y_values)
            min_y = np.min(y_values)
            df.loc[variable,'y'] = 0.5*(max_y + min_y)
            df.loc[variable,'crossbar_length'] = (max_y - min_y)
        else:
            df.loc[variable,'y'] = leaf_ordering.index(variable)
            df.loc[variable,'crossbar_length'] = 0.
    # Then walk down the tree, compiling x-positions 
    for node in tree.traverse(strategy='preorder'):
        variable = node.name
        if not node.up:
            df.loc[variable,'x'] = 0.
            continue
        df.loc[variable,'x'] = df.loc[node.up.name,'x'] + node.dist
    tree_json = df.to_json(orient='index')

    ########################################
    # 3. Write out data about the atomic time windows, which are the
    # columns of the heat map. We label the borders between columns.
    interval_endpoints = np.linspace(model.data.experiment_start,
                                     model.data.experiment_end,
                                     model.rule_population.N_intervals+1)
    # We need the timepoints later, however...
    timepoints = (0.5*(interval_endpoints[:-1] + interval_endpoints[1:]))
    time_window_edges_json = json.dumps([np.around(t,1) for t in interval_endpoints])

    ########################################
    # 4. Write out data for each cell in the heat map, simultaneously
    # compiling data about significant primitives.
    primitive_significance_threshold = 10.0
    if config.has_option('postprocessing','gui_bf_threshold'):
        primitive_significance_threshold = config.getfloat(
            'postprocessing','gui_bf_threshold'
        )
    # NB: here we map a _variable index_, as would be an attribute
    # of a primitive rule, to a leaf index, that is, a position in 
    # leaf_ordering
    variable_index_to_leaf_indices = {
        variables.index(n.name): map(
            lambda name: leaf_ordering.index(name), n.get_leaf_names()
        ) for n in model.data.variable_tree.traverse()
    }

    grid_cell_to_primitives= {} # {(i,j): [tuple1, tuple2...] ...}
    primitive_data = pd.DataFrame(
        columns=['index','bayes_factor','png','min_row','max_row',
                 'min_col','max_col','text', 'variable', 'window_start',
                 'window_end', 'type', 'direction', 'threshold', 
                 'negative_outcome_median','positive_outcome_median',
                 'beta_median', 'beta_ci_low', 'beta_ci_high']
    )
    primitive_counter = -1

    # Iterate over primitives
    for primitive, bf in summarizer.bayes_factors_by_primitive.iteritems():
        if bf < primitive_significance_threshold:
            continue
        primitive_counter += 1
        fig_name = '%d.svg' % primitive_counter
        window_start, window_stop = primitive[1]
        variable_index = primitive[0]
        primitive_rows = set()
        primitive_cols = set()
        for j,timepoint in enumerate(timepoints):
            if window_start > timepoint or window_stop < timepoint:
                continue
            primitive_cols.add(j)
            for leaf_index in variable_index_to_leaf_indices[variable_index]:
                primitive_rows.add(leaf_index)
                key = (leaf_index,j)
                grid_cell_to_primitives.setdefault(key,[]).append(
                    primitive_counter
                )
        primitive_rows = list(primitive_rows)
        primitive_cols = list(primitive_cols)
        # primitive_base is eg ((4.0, 5.5), 62, 'average')
        primitive_base = (primitive[1],primitive[0],primitive[2])
        primitive_values = (
            model.rule_population.primitive_values[primitive_base]
        )
        outcome_boolean = model.data.y.astype(bool)
        negative_median = np.median(
            primitive_values[np.logical_not(outcome_boolean)]
        )
        positive_median = np.median(
            primitive_values[(outcome_boolean)]
        )
        primitive_beta_values = summarizer.beta_by_primitive[primitive]
        beta_median = np.median(primitive_beta_values)
        beta_ci_low, beta_ci_high = np.percentile(primitive_beta_values,
                                                  (2.5,97.5))
        primitive_data.loc[primitive_counter,:] = (
            [primitive_counter,bf,fig_name,
             np.min(primitive_rows),np.max(primitive_rows),
             np.min(primitive_cols),np.max(primitive_cols),
             str(primitive), primitive[0], primitive[1][0], 
             primitive[1][1]] + list(primitive[2:]) +
            [negative_median, positive_median,
             beta_median, beta_ci_low, beta_ci_high]
        )
 
   
    primitive_data_json = primitive_data.to_json(orient='records')

    grid_columns = ['row_idx',
                    'col_idx',
                    'cell_bayes_factor',
                    'max_bayes_factor',
                    'beta',
                    'primitives']

    grid_data = pd.DataFrame(columns=grid_columns)
    counter = 0
    for i, variable in enumerate(leaf_ordering):
        for j, timepoint in enumerate(timepoints):
            cell_bf = summarizer.bayes_factors_by_grid_cell[
                (timepoint, variable)
            ]
            primitives = grid_cell_to_primitives.get((i,j),[])
            primitives = ','.join(['%d'%k for k in primitives])
            grid_data.loc[counter,:] = [i,j,cell_bf,None,None,primitives]
            counter += 1
            
    grid_data_json = grid_data.to_json(orient='records')

    ########################################
    # Output the underlying dataset, plus a title for the model
    title = 'untitled'
    if config.has_option('description', 'title'):
        title = config.get('description','title')
    elif config.has_option('description','tag'):  
        title = config.get('description','tag')

    listify_list_of_arrays = lambda l: [array.tolist() for array in l]
    to_export = {'X': listify_list_of_arrays(model.data.X),
                 'T': listify_list_of_arrays(model.data.T),
                 'y': model.data.y.astype('int').tolist(),
                 'title': title}
    dataset_json = json.dumps(to_export)

    ##################################################
    # Fill these JSON strings into the HTML template file, and save a new HTML file
    # Note, JSON strings aren't necessarily valid Javascript literals as I understand it-
    # but we ignore this complexity entirely.
    template_filename = os.path.join(os.path.dirname(__file__),
                                     'template.html')
    with open(template_filename) as f:
        html_string = f.read()
        replace_pairs = [
            (dataset_json, 'dataset_placeholder'),
            (grid_data_json, 'grid_data_placeholder'),
            (row_data_json, 'row_data_placeholder'),
            (tree_json, 'tree_placeholder'),
            (time_window_edges_json, 'time_window_edges_placeholder'),
            (primitive_data_json, 'primitive_data_placeholder')
         ]
    for json_string, inner_placeholder_string in replace_pairs:
        html_string = html_string.replace(
            '###%s###' % inner_placeholder_string,
            json_string
        )
    with open(fname,'w') as f:
        f.write(html_string)

    return [grid_data, primitive_data, tree_json, time_window_edges_json, row_data_json, dataset_json]

def tabulate_metrics(cv_results, name):
    """ Calculate accuracy metrics from probabilities, format them.

    Given a list of tuples, each of the form (index,
    vector_of_true_outcomes, vector_of_predicted_probabilities), for
    each index (representing one fold of CV) assess multiple accuracy
    metrics (eg ROC AUC, F1 score, positive predictive value) for the
    predicted probabilities WRT the true outcomes (for that fold's
    test set.) Also take the median across all folds. Then format
    these nicely into a table (labeled with the given name) and return
    that, as a string.

    For metrics which require a binary prediction, a threshold
    of 0.5 is used.
    
    """
    # Each of the metric functions should take two non-optional
    # arguments, y_true and y_pred. 
    # These accept predicted probabilities.
    probability_metrics = [
        ('AUC', roc_auc_score),
        ('AP', metrics.average_precision_score)
    ]
    # These need binary predictions
    binary_metrics = [
        ('F1', metrics.f1_score),
        ('MCC', metrics.matthews_corrcoef),
        ('precision', metrics.precision_score),
        ('recall', metrics.recall_score)
    ] 
    # Mutual information? Odds ratios?

    results = {label: [] for label, _ in
               probability_metrics + binary_metrics}
    results.update({'tn': [], 'fp': [], 'fn': [], 'tp': []})
    for label, metric in probability_metrics:
        for fold, y_true, y_pred in cv_results:
            results[label].append(metric(y_true, y_pred))
    for fold, y_true, probabilities in cv_results:
        y_pred = probabilities > 0.5
        for label, metric in binary_metrics:
            results[label].append(metric(y_true, y_pred))
        conf = zip(
            ('tn', 'fp', 'fn', 'tp'),
            metrics.confusion_matrix(y_true, y_pred).flat
        )
        for label, n in conf:
            results[label].append(n)

    index=['fold_%d' % i for i, _, _ in cv_results]
    table = pd.DataFrame(data=results, 
                         index=index)
    table.loc['median/sum'] = 0.
    for k,_ in probability_metrics + binary_metrics:
        table.loc['median/sum',k] = np.median(results[k])
    for k in ('tn', 'fp', 'fn', 'tp'):
        table.loc['median/sum',k] = np.sum(results[k])

    report = table.to_string(float_format=lambda x: '%.3g' % x)
    report = ('%s: \n' % name) + report  
    return report

