import csv
import pickle
import numpy
import numpy as np
import pandas as pd
import scipy.stats
import scipy.special
from mitre.load_data import pplacer
from mitre.rules import Dataset
from mitre import logit_rules
logger = logit_rules.logger


def run(config_parser, base_dataset):
    if config_parser.has_option('simulated_data', 'load_previous_simulation_result'):
        filename = config_parser.get('simulated_data', 'load_previous_simulation_result')
        with open(filename, 'rb') as f:
            new_dataset = pickle.load(f)

    else:
        jplace_filename = config_parser.get('data','jplace_file')
        logger.info('Processing base dataset...')
        data = dataset_interface(base_dataset, jplace_filename)
        logger.info('Reading config options...')
        config = get_config_options(config_parser)

        logger.info('Simulation begins')
        sim_data = sim_dataset(data, config)
        logger.info('Simulation ends')

        start_time = float(config['include_start_time'])
        end_time = float(config['include_end_time'])
        logger.info('Preparing simulation result')
        new_dataset = write_new_dataset(
            base_dataset,
            sim_data['samples'],
            start_time,
            end_time
        )

        if config_parser.getboolean('simulated_data','save_perturbation_info'):
            prefix = config_parser.get('description','tag')
            filename = prefix + '_perturbation.txt'
            with open(filename, 'w') as f:
                f.write('clades\n')
                f.write(str(sim_data['clades']) + '\n')
                f.write('windows\n')
                f.write(str(sim_data['time_windows']) + '\n')

    if config_parser.getboolean('simulated_data','pickle_simulation_result'):
        prefix = config_parser.get('description','tag')
        filename = prefix + '_simulated_data_object.pickle'
        logger.info('Saving simulation result to %s' % filename)
        with open(filename, 'w') as f:
            pickle.dump(new_dataset,f)

    return new_dataset

def aggregate_clade(otu_table, clade):
    return numpy.sum(otu_table[clade, :], 0)

def get_config_options(config):
    """ Reads key parameters from ConfigParser and returns dict.

    All values are returned as strings because that is what the later
    functions expect.

    """
    d = {'data_std_percent': config.get('simulated_data','data_std_percent'),
         'include_end_time': config.get('simulated_data','include_end_time'),
         'include_start_time': config.get('simulated_data','include_start_time'),
         'control_log_mean': config.get('simulated_data','control_log_mean'),
         'control_log_std': config.get('simulated_data','control_log_std'),
         'case_log_mean': config.get('simulated_data', 'case_log_mean'),
         'case_log_std': config.get('simulated_data', 'case_log_std'),
         'max_clade_abundance': config.get('simulated_data','max_clade_abundance'),
         'max_otus_in_clade': config.get('simulated_data','max_otus_in_clade'),
         'min_clade_abundance': config.get('simulated_data','min_clade_abundance'),
         'min_num_time_points_in_window': config.get('simulated_data',
                                                     'min_num_time_points_in_window'),
         'min_time_points_in_data': config.get('simulated_data',
                                               'min_time_points_in_data'),
         'time_point_std_percent': config.get(
             'simulated_data',
             'time_point_std_percent'
         ),
         'num_subjects': config.get('simulated_data','num_subjects'),
         'num_counts': config.get('simulated_data','num_counts'),
         'counts_concentration': config.get('simulated_data','counts_concentration'),
         'num_times_sim': config.get('simulated_data','num_times_sim'),
         'num_perturbations': config.get('simulated_data','num_perturbations'),
         'time_std_percent': config.get('simulated_data','time_std_percent'),
         'time_window_width': config.get('simulated_data','time_window_width'),
         'control_gets_one_pert': config.get('simulated_data', 'control_gets_one_pert')}
    return d

def sample_otu_from_posterior(config, otu_idx, otu_table, times, dt, sim_times, sim_dt, data_time_indices, limit_detection):
    # sample time-series for otu from posterior

    num_iter = 15
    # will assume time std (std per day) is scale_time_std_factor * empirical time std
    #scale_time_std_factor = 5.0

    Y = otu_table[otu_idx, :]
    Y_sim = numpy.zeros(len(sim_times))

    #time_std = numpy.asscalar(numpy.std(numpy.diff(Y) / numpy.sqrt(dt)))
    time_std = numpy.percentile(numpy.abs(numpy.diff(Y)) / numpy.sqrt(dt),75.0)

    num_time_points = len(times)
    num_sim_times = len(sim_times)

    # if the mean or time std is too small, don't attempt inference and
    # return zero values
    if (numpy.mean(Y) < limit_detection) | (time_std < limit_detection):
        ds = numpy.std(Y)
        dm = numpy.mean(Y)
        Y_sim = dm * numpy.ones(len(sim_times))
        for ni in range(len(Y_sim)):
            Y_sim[ni] = 0.0
        return Y_sim

    init_mean = numpy.asscalar(numpy.mean(Y))
    init_std = numpy.asscalar(numpy.std(Y))

    data_std_percent = float(config['data_std_percent'])

    # inflate time std
    #time_std = time_std * scale_time_std_factor

    for i in range(0, len(sim_times)):
        idx = data_time_indices[i]
        if idx < 0:
            Y_sim[i] = init_mean
        else:
            Y_sim[i] = Y[idx]

    X_sim = Y_sim.copy()

    for iter in range(0, num_iter):
        for ti in range(0, num_sim_times):
            X_sim[ti] = sample_otu_from_posterior_1step(X_sim, Y_sim, ti, sim_dt, data_time_indices, time_std,
                                                        init_mean, init_std, data_std_percent, limit_detection)
            Y_sim[ti] = sample_trunc_normal(X_sim[ti], data_std_percent*(X_sim[ti]+limit_detection), 0.0, 1.0)

    return Y_sim

def sample_otu_from_posterior_1step(X, Y, ti, dt, data_time_indices, time_std, init_mean, init_std, data_std_percent, limit_detection):
    # form proposal
    di = data_time_indices[ti]
    if ti == 0:
        ## proposal for first time-point
        if di > 0:
            v = 1.0 / (1.0 / numpy.power(init_std, 2.0) + 1.0 / numpy.power(data_std_percent*(X[0]+limit_detection), 2.0))
            m = v * (init_mean / numpy.power(init_std, 2.0) + Y[0] / numpy.power(data_std_percent*(X[0]+limit_detection), 2.0))
        else:
            v = 1.0 / (1.0 / numpy.power(init_std, 2.0))
            m = v * (init_mean / numpy.power(init_std, 2.0))
    else:
        rhs = X[ti - 1]
        ssv = numpy.power(time_std, 2.0) * dt[ti - 1]
        if di > 0:
            v = 1.0 / (1.0 / ssv + 1.0 / numpy.power(data_std_percent*(X[ti]+limit_detection), 2.0))
            m = v * (rhs / ssv + Y[ti] / numpy.power(data_std_percent*(X[ti]+limit_detection), 2.0))
        else:
            v = 1.0 / (1.0 / ssv)
            m = v * (rhs / ssv)

    val_old = X[ti]
    val_new = sample_trunc_normal(m, numpy.sqrt(v), 0.0, 1.0)

    if (val_new >= 0.0) & (val_new <= 1.0):
        l = 0
        l2 = 0

        q = 0
        q2 = 0

        p = 0
        p2 = 0

        # compute prob of forward proposal
        q = loglike_trunc_normal_1D(val_new, m, numpy.sqrt(v), 0.0, 1.0)
        # compute prob of reverse proposal
        q2 = loglike_trunc_normal_1D(val_old, m, numpy.sqrt(v), 0.0, 1.0)

        if di > 0:
            # compute data prob under forward proposal
            l = loglike_trunc_normal_1D(Y[ti], val_new, data_std_percent*(X[ti]+limit_detection), 0.0, 1.0)
            # compute data prob under reverse proposal
            l2 = loglike_trunc_normal_1D(Y[ti], val_old, data_std_percent*(X[ti]+limit_detection), 0.0, 1.0)

        # compute trajectory probs under proposal
        if ti == 0:
            p = loglike_trunc_normal_1D(val_new, init_mean, init_std, 0.0, 1.0)
            p2 = loglike_trunc_normal_1D(val_old, init_mean, init_std, 0.0, 1.0)
        else:
            p = loglike_trunc_normal_1D(val_new, X[ti - 1], time_std * numpy.sqrt(dt[ti - 1]), 0.0, 1.0)
            p2 = loglike_trunc_normal_1D(val_old, X[ti - 1], time_std * numpy.sqrt(dt[ti - 1]), 0.0, 1.0)

        # if it's not the last time-point, then it affects the next time-point
        if ti < ti - 1:
            p = p + loglike_trunc_normal_1D(X[ti + 1], val_new, time_std * numpy.sqrt(dt[ti]), 0.0, 1.0)
            p2 = p2 + loglike_trunc_normal_1D(X[ti + 1], val_old, time_std * numpy.sqrt(dt[ti]), 0.0, 1.0)

        ## now calculate accept ratio
        r = -(p2 + l2) + (p + l) - (q - q2)
        r = numpy.min([1, numpy.exp(r)])

        if numpy.random.uniform() < r:
            if not numpy.isnan(val_new):
                return val_new

    return val_old

def sample_from_prior(config, times, dt, time_window, outcome, limit_detection):
    control_log_mean = float(config['control_log_mean'])
    control_log_std = float(config['control_log_std'])

    case_log_mean = float(config['case_log_mean'])
    case_log_std = float(config['case_log_std'])

    time_std = float(config['time_std_percent']) * numpy.exp(control_log_mean)
    data_std_percent = float(config['data_std_percent'])

    perturb_on_time = time_window[0]
    perturb_off_time = time_window[1]

    num_time_points = len(times)

    X = numpy.zeros(num_time_points)
    Y = numpy.zeros(num_time_points)

    perturb = numpy.exp(numpy.random.normal(case_log_mean,case_log_std))
    if perturb > 1.0:
        perturb = 0.99

    perturb_on = False
    for ti in range(0, num_time_points):
        if ti == 0:
            X[0] = numpy.exp(numpy.random.normal(control_log_mean,control_log_std))
            if X[0] > 1.0:
                X[0] = 0.99
        else:
            if outcome is True:
                if (perturb_on is False) & (times[ti] >= perturb_on_time):
                    perturb_on = True
                if (perturb_on is True) & (times[ti] > perturb_off_time):
                    perturb_on = False
                    X[ti - 1] = X[0]

            if (perturb_on is True):
                X[ti] = perturb
            else:
                X[ti] = sample_trunc_normal(X[ti - 1], time_std * numpy.sqrt(dt[ti - 1]), 0.0, 1.0)

        Y[ti] = sample_trunc_normal(X[ti], data_std_percent*(X[ti]+limit_detection), 0.0, 1.0)

    return Y

def sample_trunc_normal(my_mean, my_std, myclip_a, myclip_b):
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    return scipy.stats.truncnorm.rvs(a, b, loc=my_mean, scale=my_std)

def loglike_trunc_normal_1D(data, my_mean, my_std, myclip_a, myclip_b):
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    return scipy.stats.truncnorm.logpdf(data, a, b, loc=my_mean, scale=my_std)

def make_data_time_indices(times, sim_times):
    # sets up correspondence between actual sampled times, and times for simulation
    num_time_points = len(times)
    num_sim_times = len(sim_times)
    data_time_indices = numpy.zeros(num_sim_times, dtype=int)

    for i in range(0, num_sim_times):
        data_time_indices[i] = -1
        for j in range(0, num_time_points):
            if sim_times[i] == times[j]:
                data_time_indices[i] = j

    return data_time_indices

def resample_otus(subjID, otu_table, config, clades, times, sim_times, time_windows, outcome):
    limit_detection = 1.0 / 10000.0

    dt = numpy.diff(times)
    sim_dt = numpy.diff(sim_times)

    num_otus = numpy.shape(otu_table)[0]
    resampled_otus = numpy.zeros((num_otus, len(sim_times)))

    data_time_indices = make_data_time_indices(times, sim_times)

    for o in range(0, num_otus):
        #if o % 100 == 0:
        #    print(o, end=' ', flush=True)
        resampled_otus[o, :] = sample_otu_from_posterior(config, o, otu_table, times, dt, sim_times, sim_dt, data_time_indices,
                                                         limit_detection)

    #print("\n")

    # compute proportions of clade members
    clade_props = []
    for ci in range(0, len(clades)):
        clade = clades[ci]
        clade_prop = resampled_otus[clade, :].copy()
        total_clade = numpy.sum(clade_prop, 0)
        clade_prop = clade_prop / total_clade
        clade_props.append(clade_prop)

        # take out values, so we can inject simulated clade
        for o in clade:
            resampled_otus[o, :] = resampled_otus[o, :] * 0

    # now renormalize
    total_counts = numpy.sum(resampled_otus, axis=0)
    resampled_otus = resampled_otus / total_counts

    total_perturb_clade = numpy.zeros(len(sim_times))
    perturb_clades = []

    # if more than 1 clade to perturb, and it's a control,
    # then deterministically pick 1 of the clades to perturb
    if not outcome:
        rand_perturb_clade = 0

    for ci in range(0, len(clades)):
        perturb_clade = sample_from_prior(config, sim_times, sim_dt, time_windows[ci], outcome, limit_detection)
        if config['control_gets_one_pert'] == 'True':
            if (len(clades) > 1) & (not outcome):
                if ci == rand_perturb_clade:
                    perturb_clade = sample_from_prior(config, sim_times, sim_dt, time_windows[ci], True, limit_detection)
        total_perturb_clade = total_perturb_clade + perturb_clade
        perturb_clades.append(perturb_clade)

    # adjust proportions of other otus to accomodate simulated clades
    resampled_otus = resampled_otus * (1 - total_perturb_clade)

    # now inject simulated clades
    for ci in range(0, len(clades)):
        clade_prop = clade_props[ci]
        perturb_clade = perturb_clades[ci]
        resampled_otus[clades[ci], :] = numpy.asarray(clade_prop) * numpy.asarray(perturb_clade)

    # final renormalize
    total_counts = numpy.sum(resampled_otus, axis=0)
    resampled_otus = resampled_otus / total_counts

    #return resampled_otus
    return ra_to_counts(resampled_otus, config)

def ra_to_counts(relative_abundances, config):
    """ Simulate counts from relative abundnace matrix.

    Returns a matrix of the same shape as relative_abundances.
    Each column of the return matrix is drawn from a Dirichlet multinomial
    distribution with number of trials config['num_counts']
    and expected frequency parameter taken from the corresponding
    column of the input matrix, with a concentration parameter
    config['counts_concentration'].

    """
    assert np.allclose(np.sum(relative_abundances, 0), 1.0)

    concentration = float(config['counts_concentration'])
    trials = int(config['num_counts'])

    rows, columns = relative_abundances.shape
    counts = np.zeros((rows,columns),dtype=int)
    for j in xrange(columns):
        this_sample_ra = relative_abundances[:,j]
        nonzero_indices = np.where(this_sample_ra>0.)[0]
        nonzero_values = this_sample_ra[this_sample_ra>0]
        alpha = concentration * nonzero_values
        f = np.random.dirichlet(alpha)
        v = np.random.multinomial(trials, f)
        counts[nonzero_indices,j] = v

    return counts #, nonzero_values, alpha, f, concentration, v


def read_config(fname):
    ct = readtable_csv(fname)
    config = {}
    for i in range(0, len(ct)):
        config[ct[i][0]] = ct[i][1]

    return config


def dataset_interface(dataset, jplace_file):
    """ Format data from a MITRE dataset object for simulation code

    Assume the dataset contains raw data (in particular, it has not
    been converted to a relative abundance scale, nor aggregated on
    the tree yet- this is why jplace_file must be passed.)

    """
    # subject_abundance_dict: {subject id: basically X[i]- n_otus x
    # subject_times_dict: {subject id: sorted list of timepoints}
    # control_subjects a list of sample ids
    # clades a list of lists of indices
    subject_abundance_dict = dict(zip(dataset.subject_IDs,
                                      dataset.X))
    control_subjects = [s for s, outcome in
                        zip(dataset.subject_IDs, dataset.y) if
                        not outcome]
    subject_times_dict = dict(zip(dataset.subject_IDs,dataset.T))
    for v in subject_times_dict.values():
        assert tuple(v) == tuple(sorted(v))
    with_tree, _, _ = pplacer.aggregate_by_pplacer_simplified(
        jplace_file,
        dataset
    )
    clades = [
        [dataset.variable_names.index(i) for i in node.get_leaf_names()] for
        node in with_tree.variable_tree.iter_descendants()
    ]
    # Match Georg's original implementation: consider only clades with
    # at least 2 OTUs
    clades = [c for c in clades if len(c) > 1]

    return {'subject_abundance_dict': subject_abundance_dict,
            'subject_times_dict': subject_times_dict,
            'clades': clades,
            'control_subjects': control_subjects}

def read_data(basedir):
    sample_metadata = readtable_csv(basedir + 'sample_metadata.csv')
    abundance = readtable_csv(basedir + 'abundance.csv')
    subject_data = readtable_csv(basedir + 'subject_data.csv')
    clades_t = readtable_csv(basedir + 'clades.csv')
    control_subjects_t = readtable_csv(basedir + 'control_subjects.csv')
    control_subjects = [y for x in control_subjects_t for y in x]

    clades = []
    for c in clades_t:
        ct = []
        for cv in c:
            ct.append(int(cv) - 1)
        clades.append(ct)

    num_otus = len(abundance[0]) - 1

    # create dictionary for sample lookup
    sample_dict = {}
    for i in range(1, len(abundance)):
        sample_dict[abundance[i][0]] = i

    # create dictionaries for subjects, mapping time-points to samples
    subject_tp_dict = {}
    for i in range(0, len(sample_metadata)):
        subj_ID = sample_metadata[i][1]
        if not subj_ID in subject_tp_dict:
            subject_tp_dict[subj_ID] = {}

        subj_dict = subject_tp_dict[subj_ID]
        exp_ID = sample_metadata[i][0]

        # make sure data is actually present
        if exp_ID in sample_dict:
            subj_dict[int(sample_metadata[i][2])] = exp_ID

    # now generate matrices for each subject
    subject_abundance_dict = {}
    subject_times_dict = {}
    for subj_ID, subj_dict in sorted(subject_tp_dict.items()):
        subj_times = []
        data_matrix = numpy.zeros((num_otus, len(subj_dict)))
        ti = 0
        for t, exp_ID in sorted(subj_dict.items()):
            subj_times.append(t)
            sample_idx = sample_dict[exp_ID]
            dv = numpy.array(abundance[sample_idx][1:])
            data_matrix[:, ti] = dv
            ti = ti + 1
        subject_times_dict[subj_ID] = subj_times
        subject_abundance_dict[subj_ID] = data_matrix

    return {'subject_abundance_dict': subject_abundance_dict,
            'subject_times_dict': subject_times_dict,
            'subject_data_dict': subject_data_dict,
            'subject_data_col_dict': subject_data_col_dict, 'clades':
            clades, 'control_subjects': control_subjects}


def trim_data(D, sim_times, interp_times):
    # return matrix with interpolated times only
    num_otus = numpy.shape(D)[0]
    D2 = numpy.zeros((num_otus, len(interp_times)))
    for ti in range(0, len(interp_times)):
        tix = numpy.where(sim_times == interp_times[ti])[0][0]
        D2[:, ti] = D[:, tix]

    return D2

def generate_noisy_interp_times(config):
    start_time = float(config['include_start_time'])
    end_time = float(config['include_end_time'])
    num_times = int(config['num_times_sim'])
    typical_spacing = (end_time-start_time)/num_times
    std = float(config['time_point_std_percent'])*typical_spacing
    t = numpy.linspace(start=start_time,
                     stop=end_time,
                     endpoint=False,
                     num=num_times) + 0.5*typical_spacing
    delta = std * np.random.randn(num_times)
    # Tweak this a bit to reduce the likelihood of pathological cases
    delta = np.fmax(delta,-0.249*typical_spacing)
    delta = np.fmin(delta,0.249*typical_spacing)
    # Don't move the start/end times outside the specified window
    if delta[0] < 0:
        delta[0] = 0
    if delta[-1] > 0:
        delta[-1] = 0
    sim_times = np.sort(t + delta)
    return sim_times

def generate_all_good_windows(config):
    start_time = float(config['include_start_time'])
    end_time = float(config['include_end_time'])
    window_width = float(config['time_window_width'])

    good_windows = []

    for t in numpy.arange(start_time + 1.0, end_time - window_width - 1.0):
        good_windows.append([t, t + window_width])

    return good_windows


def sim_subject(data, config, clades, num_times_sim, perturb_params, subjID, outcome):
    otu_table = data['subject_abundance_dict'][subjID]
    times = data['subject_times_dict'][subjID]

    # for simulating, add interpolated times to existing time-points
    interp_times = generate_noisy_interp_times(config)
    sim_times = numpy.union1d(times, interp_times)

    total_counts = numpy.sum(otu_table, 0)
    # convert to percentages
    otu_table = otu_table.astype(float) / total_counts.astype(float)

    resampled_otus = resample_otus(subjID, otu_table, config, clades, times,
                                   sim_times, perturb_params, outcome)

    # discard simulated data outside the evenly spaced time-points
    resampled_otus = trim_data(resampled_otus, sim_times, interp_times)
    sim_times = interp_times

    return {'otu_table': otu_table, 'resampled': resampled_otus,
            'times': times, 'sim_times': sim_times, 'subjID': subjID,
            'outcome': outcome}


def generate_clade_abundances(data, subjects):
    num_subjects = len(subjects)
    clades = data['clades']
    num_clades = len(clades)

    clade_abundances = numpy.zeros((num_clades, num_subjects))

    for si in range(0, len(subjects)):
        subjID = subjects[si]
        d = data['subject_abundance_dict'][subjID]

        total_counts = numpy.sum(d, 0)
        # convert to percentages
        d = d.astype(float) / total_counts.astype(float)
        for ci in range(0, num_clades):
            clade = clades[ci]
            clade_abundances[ci, si] = numpy.max(aggregate_clade(d, clade))

    return clade_abundances


def filter_subjects(data, include_start_time, include_end_time, min_no_times):
    filtered_subjects = []
    subjects = data['control_subjects']

    for s in range(0, len(subjects)):
        subjID = subjects[s]
        times = data['subject_times_dict'][subjID]
        if len(times) >= min_no_times:
            if ((numpy.min(times) <= include_start_time) &
                (numpy.max(times) >= include_end_time)):
                filtered_subjects.append(subjID)

    return filtered_subjects


def find_clades(min_p, max_p, max_in_clade, data, subjects):
    clade_abundances = generate_clade_abundances(data, subjects)
    max_ca = numpy.max(clade_abundances, 1)
    min_ca = numpy.min(clade_abundances, 1)

    good_clades = []
    for i in range(len(data['clades'])):
        if ((max_ca[i] <= max_p) & (min_ca[i] >= min_p) &
            (len(data['clades'][i]) <= max_in_clade)):
            good_clades.append(i)

    return good_clades


def sim_dataset(data, config):
    num_subjects = int(config['num_subjects'])
    num_perturbations = int(config['num_perturbations'])
    if num_perturbations not in (1,2):
        raise ValueError()

    num_times_sim = int(config['num_times_sim'])
    time_window_width = int(config['time_window_width'])
    min_num_time_points_in_window = int(config['min_num_time_points_in_window'])

    min_time_points_in_data = int(config['min_time_points_in_data'])
    include_start_time = int(config['include_start_time'])
    include_end_time = int(config['include_end_time'])

    min_clade_abundance = float(config['min_clade_abundance'])
    max_clade_abundance = float(config['max_clade_abundance'])
    max_otus_in_clade = int(config['max_otus_in_clade'])

    filtered_subjects = filter_subjects(data, include_start_time,
                                        include_end_time, min_time_points_in_data)
    logger.info(str(len(filtered_subjects)) + " subjects selected")
    good_clades = find_clades(
        min_clade_abundance,
        max_clade_abundance, max_otus_in_clade, data, filtered_subjects
    )
    good_windows = generate_all_good_windows(config)

    # randomly sample a clade and a time window
    if num_perturbations == 1:
        rand_clade = [
            data['clades'][good_clades[numpy.random.randint(len(good_clades))]]
        ]
        rand_window = [good_windows[numpy.random.randint(len(good_windows))]]
    else:
        # Note that because of the way the data is generated
        # we don't want one clade to contain the other- if the
        # time windows overlap, data for shared OTUs would be overwritten,
        # rather than receiving two perturbations
        overlapping = True
        while overlapping:
            rand_clade = [data['clades'][i] for i in
                          np.random.choice(good_clades,2,replace=False)]
            if not set(rand_clade[0]).intersection(rand_clade[1]):
                overlapping = False
                logger.info('Confirmed target clades do not overlap')
            else:
                logger.info('Clades overlap, retrying')
        # can't use random.choice on good_windows, it looks like a
        # 2D array
        window_i, window_j = np.random.randint(0, len(good_windows), 2)
        rand_window = [good_windows[window_i], good_windows[window_j]]

    logger.info('Perturbing clade(s) %s' % str(rand_clade))
    logger.info('Applying perturbation in window(s) %s' % str(rand_window))

    samples = []

    for s in range(0, num_subjects):
        # sample from actual subjects with replacement
        sn = numpy.random.randint(len(filtered_subjects))
        subjID = filtered_subjects[sn]
        outcome = False
        if s % 2 == 0:
            outcome = True

        logger.info('Simulating subject %d based on subject id %s with outcome %d' %
                    (s, str(subjID), int(outcome)))
        samples.append(sim_subject(data, config, rand_clade, num_times_sim, rand_window, subjID, outcome))

    return {'samples': samples, 'clades': rand_clade, 'time_windows': rand_window}

def write_new_dataset(base_dataset, sim_data, start_time, end_time):
    # sim_data is the 'samples' return value of sim_dataset
    # It is a list of output dictionaries of the form:
    # {'otu_table': otu_table, 'resampled': resampled_otus, 'times': times,
    #  'sim_times': sim_times, 'subjID': subjID, 'outcome':
    #  outcome}
    # where resampled_otus is a data matrix, outcome is a boolean, and
    # sim_times are the times corresponding to the columns of resampled_otus;
    # subjID is the subject ID used as a basis for simulating this subject,
    # not an ID for this subject.
    new_dataset = base_dataset.copy()
    X = []
    y = []
    T = []
    ids = []
    for i,sample in enumerate(sim_data):
        ids.append('sim%d' % i)
        X.append(sample['resampled'])
        T.append(sample['sim_times'])
        y.append(sample['outcome'])
    subject_data = pd.DataFrame(index=ids)
    subject_data['outcome'] = y
    return Dataset(
        X=X, T=T, y=y,
        variable_names = base_dataset.variable_names,
        variable_weights = base_dataset.variable_weights,
        experiment_start = start_time,
        experiment_end = end_time,
        subject_IDs = ids,
        subject_data = subject_data,
        variable_annotations = base_dataset.variable_annotations
    )
