# Written by Matthew Honnibal
# http://honnibal.wordpress.com
#
import random
from cymem.cymem cimport Pool

from libc.math cimport sqrt

cimport cython

ctypedef double[2] double_pair

cdef class World:
    cdef Pool mem
    cdef int N
    cdef double* m
    cdef double_pair* r
    cdef double_pair* v
    cdef double_pair* F
    cdef readonly double dt
    def __init__(self, N, threads=1, m_min=1, m_max=30.0, r_max=50.0, v_max=4.0, dt=1e-3):
        self.mem = Pool()
        self.N = N
        #m = np.random.uniform(m_min, m_max, N)
        #r = np.random.uniform(-r_max, r_max, (N, 2))
        # v = np.random.uniform(-v_max, v_max, (N, 2))
        #F  = np.zeros_like(self.r)
        self.m = <double*>self.mem.alloc(N, sizeof(double))
        self.r = <double_pair*>self.mem.alloc(N, sizeof(double_pair))
        self.v = <double_pair*>self.mem.alloc(N, sizeof(double_pair))
        self.F = <double_pair*>self.mem.alloc(N, sizeof(double_pair))
        for i in range(N):
            self.m[i] = random.uniform(m_min, m_max)
            for j in range(2):
                self.r[i][j] = random.uniform(-r_max, r_max)
                self.v[i][j] = random.uniform(-v_max, v_max)
                self.F[i][j] = 0
        self.dt = dt


@cython.cdivision(True)
def compute_F(World w):
    """Compute the force on each body in the world, w."""
    cdef int i, j
    # Set all forces to zero. 
    for i in range(w.N):
        w.F[i][0] = 0
        w.F[i][1] = 0
    cdef double sx, sy, s3, tmp, Fx, Fy
    for i in range(w.N):
        for j in range(i+1, w.N):
            sx = w.r[j][0] - w.r[i][0];
            sy = w.r[j][1] - w.r[i][1];

            s3 = sqrt(sx*sx + sy*sy);
            s3 *= s3 * s3;

            tmp = w.m[i] * w.m[j] / s3;
            Fx = tmp * sx;
            Fy = tmp * sy;

            w.F[i][0] += Fx;
            w.F[i][1] += Fy;

            w.F[j][0] -= Fx;
            w.F[j][1] -= Fy;


@cython.cdivision(True)
def evolve(World w, int steps):
    """Evolve the world, w, through the given number of steps."""
    cdef int _, i
    for _ in range(steps):
        compute_F(w)
        for i in range(w.N):
            w.v[i][0] += w.F[i][0] * w.dt / w.m[i]
            w.v[i][1] += w.F[i][1] * w.dt / w.m[i]
            w.r[i][0] += w.v[i][0] * w.dt
            w.r[i][1] += w.v[i][1] * w.dt
