#cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as np
from scipy.linalg import blas, solve_triangular
cimport scipy.linalg.cython_blas
from libc.math cimport log
from libc.math cimport abs as c_abs


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cimport cython

def likelihoods_of_all_options(
    np.ndarray[DTYPE_t, ndim=2] X,
    np.ndarray[DTYPE_t, ndim=1] omega,
    np.ndarray[DTYPE_t, ndim=1] response,
    np.ndarray[DTYPE_t, ndim=2] list_of_truth_vectors,
    DTYPE_t prior_coefficient_variance
    ):
    """ Evaluate MVN logpdf after adding every possible rule to X. 

    """
    cdef DTYPE_t sqrt_prior_coefficient_variance = np.sqrt(
        prior_coefficient_variance
    )

    cdef int number_of_options = len(list_of_truth_vectors)
    cdef int k = len(list_of_truth_vectors[0])
    cdef np.ndarray[DTYPE_t, ndim=1] likelihoods = np.zeros(
        len(list_of_truth_vectors),
        dtype=DTYPE
    )

    cdef np.ndarray[DTYPE_t, ndim=1] z = response

    cdef np.ndarray[DTYPE_t, ndim=2] base_covariance_z = (
        np.diag(1./omega) + (prior_coefficient_variance * 
                             np.dot(X,X.T))
    )
    cdef np.ndarray[DTYPE_t, ndim=2] base_R = np.linalg.cholesky(base_covariance_z)
    cdef np.ndarray[DTYPE_t, ndim=2] base_R_T = base_R.T
    cdef np.ndarray[DTYPE_t, ndim=1] base_y = solve_triangular(base_R,z,
                              lower=True,check_finite=False)

    normalization = 0.5*k*np.log(2.0*np.pi)

    cdef np.ndarray[DTYPE_t, ndim=2] R_update_top_row_states = np.zeros((k+1,k))
    cdef np.ndarray[DTYPE_t, ndim=1] R_update_diagonal_elements = np.zeros(k)
    cdef np.ndarray[DTYPE_t, ndim=1] R_update_diagonal_log_square_partial_sums = np.zeros(k+1)
    cdef np.ndarray[DTYPE_t, ndim=1] y_update_pseudo_element_values = np.zeros(k+1)
    cdef np.ndarray[DTYPE_t, ndim=1] y_dot_y_partial_sums = np.zeros(k+1)
    
    cdef np.ndarray[DTYPE_t, ndim=1] givens_cosines = np.zeros(k)
    cdef np.ndarray[DTYPE_t, ndim=1] givens_sines = np.zeros(k)
    cdef int option_index, divergence_index
    cdef np.ndarray[DTYPE_t, ndim=1] v = np.zeros(k)
    cdef np.ndarray[DTYPE_t, ndim=1] last_v = np.zeros(k)
    cdef int i
    cdef int catchup_row, catchup_column, update_index
    cdef DTYPE_t current_givens_s, current_givens_c
    cdef DTYPE_t this_entry_new_solution, log_det_cov, y_dot_y
    cdef DTYPE_t drotg_argument_a, drotg_argument_b

    for option_index in xrange(number_of_options):
        #print 'option_index %d' % option_index
        for i in xrange(k):
            v[i] = list_of_truth_vectors[option_index,i]
        if option_index == 0:
            divergence_index = 0
        else:
            divergence_index = k
            for i in xrange(k):
                if v[i] != last_v[i]:
                    divergence_index = i
                    break
        #print divergence_index
        for i in xrange(k):
            last_v[i] = v[i]

        for i in xrange(divergence_index, k):
            R_update_top_row_states[0,i] = sqrt_prior_coefficient_variance * v[i]
        ##print 'post-divergence update'
        ##print R_update_top_row_states[0,:]

        # Run the columns of v at divergence_index and rightward
        # through the first divergence_index - 1 givens rotations,
        # updating those columns in the rows at divergence_index 
        # and greater in the running log of 'top rows'
        # May be writing one too many columns worth of entries here
        for catchup_column in xrange(divergence_index, k):
            for catchup_row in xrange(divergence_index):
                R_update_top_row_states[catchup_row+1,
                                        catchup_column] = (
                    (givens_cosines[catchup_row] * 
                     R_update_top_row_states[catchup_row, 
                                             catchup_column]) -
                    (givens_sines[catchup_row] *
                     base_R_T[catchup_row, catchup_column])
                )
            
        # Now continue with the calculation from the divergence point
        for update_index in xrange(divergence_index, k):
            # Use temporary variables as cython_blas.drotg 
            # modifies the values to which its inputs point
            drotg_argument_a = base_R_T[update_index, update_index]
            drotg_argument_b = R_update_top_row_states[update_index, update_index]
            scipy.linalg.cython_blas.drotg(
                &drotg_argument_a,
                &drotg_argument_b,
                &current_givens_c,
                &current_givens_s
            )
            givens_cosines[update_index] = current_givens_c
            givens_sines[update_index] = current_givens_s

            for i in xrange(update_index,k):
                R_update_top_row_states[update_index+1, i] = (
                    (current_givens_c *
                     R_update_top_row_states[update_index, i]) -
                    (current_givens_s * base_R_T[update_index, i])
                 )

            R_update_diagonal_elements[update_index] = (
                (current_givens_s *
                 R_update_top_row_states[update_index,update_index]) +
                (current_givens_c * 
                 base_R_T[update_index, update_index])
            )
            R_update_diagonal_log_square_partial_sums[update_index+1] = (
                R_update_diagonal_log_square_partial_sums[update_index] 
                + 2.0*log(c_abs(R_update_diagonal_elements[update_index]))
            )
            
            y_update_pseudo_element_values[update_index+1] = (
                current_givens_c * y_update_pseudo_element_values[update_index] -
                current_givens_s * base_y[update_index]
            )
            this_entry_new_solution = (
                current_givens_s * y_update_pseudo_element_values[update_index] +
                current_givens_c * base_y[update_index]
            )
            y_dot_y_partial_sums[update_index + 1] = (
                y_dot_y_partial_sums[update_index] + 
                this_entry_new_solution * this_entry_new_solution 
            )
        #print np.around(R_update_top_row_states,3)[:10,:10]
        #print 'givens cosines'
        #print givens_cosines
        #print 'givens sines'
        #print givens_sines

        #print R_update_diagonal_log_square_partial_sums
        #print y_dot_y_partial_sums
            
        #Get the actual likelihood
        #print 'diag R:'
        #print R_update_diagonal_elements
        log_det_cov = R_update_diagonal_log_square_partial_sums[k]
        #print log_det_cov
        y_dot_y = y_dot_y_partial_sums[k]
        #print y_dot_y
        #We collect the likelihoods unnormalized and off by a factor
        #of -2
        
        likelihoods[option_index] = y_dot_y + log_det_cov
        
    return -0.5 * likelihoods - normalization


