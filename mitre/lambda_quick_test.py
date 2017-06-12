import numpy as np
from scipy.stats import gamma
from scipy.stats import poisson

hyper_a = 25./10000.
hyper_b = 100./10000.
mean_lambda = hyper_a/hyper_b

def draw(N):
    lambdas = []
    thetas = []
    this_lambda = mean_lambda
    for i in xrange(N):
        theta = poisson.rvs(this_lambda)
        new_alpha = (hyper_a + theta)
        new_beta = (hyper_b + 1.0)
        # Note scipy.stats.gamma uses a shape and scale parameterization
        this_lambda = gamma.rvs(new_alpha, scale=1.0/new_beta)
        lambdas.append(this_lambda)
        thetas.append(theta)
    return lambdas, thetas

        
