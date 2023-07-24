import cython
from libc.math cimport atan2


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double kay_weighted_complex(double[:] R, double[:] I, double[:] W):
    # initialize variables
    cdef Py_ssize_t N = R.shape[0]
    cdef Py_ssize_t i = 0
    cdef double f_est = 0

    # main loop
    for i in range(N - 1):
        f_est += W[i] * atan2(R[i]*I[i + 1] - I[i]*R[i + 1],
                              R[i]*R[i + 1] + I[i]*I[i + 1])

    return f_est


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int abs_argmax(double[:] R, double[:] I):
    # initialize variables
    cdef Py_ssize_t N = R.shape[0]
    cdef Py_ssize_t i = 0
    cdef int max_idx = 0
    cdef double current_max = 0
    cdef double current_abs2 = 0

    # main loop
    for i in range(N):
        current_abs2 = R[i]*R[i] + I[i]*I[i]
        if current_abs2 > current_max:
            max_idx = i
            current_max = current_abs2

    return max_idx
