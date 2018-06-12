import csv
import numpy
import os
from argparse import ArgumentParser

def generate_diagnostics(list_of_filenames):
    (col_names, runs) = read_mcmc_traces(list_of_filenames)
    runs2 = divide_runs(runs)

    num_samples = numpy.shape(runs2[0])[0]

    R_hat = []

    for n in range(250,num_samples+1,250):
        R_hat.append(compute_R_hat(runs2,n-1))
    header = ','.join(['iterations'] + col_names[1:])
    print header

    idx=0
    for n in range(250, num_samples + 1, 250):
        row = ('%s,' % n) + ','.join(map(str,R_hat[idx].T))
        print row
        idx += 1

def read_mcmc_traces(list_of_filenames):
    f=open(list_of_filenames[0])
    reader = csv.reader(f)
    col_names = next(reader)
    f.close()
    runs = []

    for fname in list_of_filenames:
        d = numpy.genfromtxt(fname, delimiter=",", skip_header=1)
        runs.append(d)

    return (col_names,runs)

def divide_runs(runs):
    # split runs in half
    runs2 = []

    for i in range(0,len(runs)):
        num_samples = numpy.shape(runs[i])[0]
        num_cols = numpy.shape(runs[i])[1]
        spl = int(numpy.floor(num_samples/2))

        s1 = runs[i][0:(spl-1),:]
        s1 = s1[:,1:num_cols]
        runs2.append(s1)

        s2 = runs[i][spl:num_samples, :]
        s2 = s2[:,1:num_cols]
        runs2.append(s2)

    return runs2

def compute_R_hat(runs2,n):
    # compute the R_hat measure (Gelman et al, Bayesian Data Analysis 3rd Ed, CRC Press, p284-285)

    m = len(runs2)
    nv = numpy.shape(runs2[0])[1]

    pj = numpy.zeros((m,nv))
    sj = numpy.zeros((m,nv))

    for j in range(0,m):
        pj[j,:] = numpy.mean(runs2[j][0:n,:],axis=0)

    for j in range(0,m):
        sj[j,:] = numpy.sum(numpy.power(runs2[j][0:n, :] - pj[j,:],2.0),axis=0)/(n-1)

    B = numpy.zeros(nv)
    pp = numpy.mean(pj,axis=0)
    for j in range(0,m):
        B = B + numpy.power(pj[j]-pp,2.0)
    B = B*n/(m-1)

    W = numpy.mean(sj,axis=0)

    var = W*(n-1)/n + B/n

    R_hat = numpy.sqrt(var/W)

    return R_hat

def run():
    parser = ArgumentParser(
        description="calculate Gelman's R-hat measure from output files from multiple MITRE MCMC runs",
        epilog='Results are printed to standard output.'
    )
    parser.add_argument('input_files',metavar='mcmc_trace_file',nargs='+',
                        help='paths to MITRE MCMC trace CSV files')
    args = parser.parse_args()
    generate_diagnostics(args.input_files)
