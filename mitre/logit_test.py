import numpy as np
np.random.seed(751381789)
import logit

#X = np.ones((100,3))
#X[:50,0] = 0
#X[::2,1] = 0.
#beta = np.array([4,2,-3])

X = np.ones((100,2))
X[:50,0] = 0
beta = np.array([4,-3])

psi = np.dot(X,beta)
p = 1./(1.+np.exp(-1.0*psi))

results = []
for i in xrange(20):
    true_y = np.random.rand(100) < p
    state0 = logit.LogisticRegressionState(X,true_y,100,np.zeros(len(beta)))
    state,betas,omegas = logit.sample(state0,5000)
    betas = zip(*betas)
    results.append([np.mean(b) for b in betas])
