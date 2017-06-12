"""
Calculate the multivariate normal PDF using Cholesky decomposition or other methods.

Somewhat dangerously, all check_finite arguments to linear algebra routines are 'False' here.


"""
import numpy as np
import scipy.linalg 

def mc_logpdf(x, cov):
    """
    Calculate the multivariate normal PDF using Cholesky decomposition.
    
    This should be faster than the eigenvalue-decomposition based approach
    of the scipy implementation, though not as thoroughly tested,
    and not robust to zero eigenvalues of cov (not usually an issue 
    for us.)

    """
    k, _ = cov.shape
#    R = np.linalg.cholesky(cov)
    R = scipy.linalg.cholesky(cov,check_finite=False)
    log_det_cov = np.sum(np.log(R.diagonal()))*2.0
#    det_cov = np.prod(R.diagonal())**2.0
    y = scipy.linalg.solve_triangular(R.T,x,lower=True,check_finite=False)
    exponent = -0.5*np.sum(np.square(y))
#    print exponent
#    print log_det_cov
    return exponent - 0.5*(k*np.log(2.0*np.pi)+log_det_cov)


def mc_logpdf_finite(x, cov):
    """
    Calculate the multivariate normal PDF using Cholesky decomposition.
    
    This should be faster than the eigenvalue-decomposition based approach
    of the scipy implementation, though not as thoroughly tested,
    and not robust to zero eigenvalues of cov (not usually an issue 
    for us.)

    """
    k, _ = cov.shape
#    R = np.linalg.cholesky(cov)
    R = scipy.linalg.cholesky(cov,check_finite=True)
    det_cov = np.prod(R.diagonal())**2.0
    y = scipy.linalg.solve_triangular(R,x,lower=True,check_finite=False)
    exponent = -0.5*np.sum(np.square(y))
    print exponent
    print det_cov
    return exponent - 0.5*(k*np.log(2.0*np.pi)+np.log(det_cov))

def mc_logpdf_np(x, cov):
    """
    Calculate the multivariate normal PDF using Cholesky decomposition.
    
    This should be faster than the eigenvalue-decomposition based approach
    of the scipy implementation, though not as thoroughly tested,
    and not robust to zero eigenvalues of cov (not usually an issue 
    for us.)

    This version uses the numpy, not scipy, Cholesky factorization routine.

    """
    k, _ = cov.shape
    R = np.linalg.cholesky(cov)
#    R = scipy.linalg.cholesky(cov,check_finite=False)
#    det_cov = np.prod(R.diagonal())**2.0
    log_det_cov = np.sum(np.log(R.diagonal()))*2.0
    y = scipy.linalg.solve_triangular(R,x,lower=True,check_finite=False)
    exponent = -0.5*np.sum(np.square(y))
#    print det_cov
#    print exponent
    return exponent - 0.5*(k*np.log(2.0*np.pi)+log_det_cov)

def qr_logpdf(x, cov):
    """
    Calculate the multivariate normal PDF using QR decomposition.
    
    This should be faster than the eigenvalue-decomposition based approach
    of the scipy implementation, though not as thoroughly tested,
    and not robust to zero eigenvalues of cov (not usually an issue 
    for us.)

    """
    k, _ = cov.shape
    Q,R = scipy.linalg.qr(cov,check_finite=False)
    det_cov = np.prod(R.diagonal())
    QTx = np.dot(Q.T,x)
    y = scipy.linalg.solve_triangular(R.T,x,lower=True,check_finite=False)
    exponent = -0.5*np.dot(y,QTx)
#    print exponent
    return exponent - 0.5*(k*np.log(2.0*np.pi)+np.log(det_cov))
