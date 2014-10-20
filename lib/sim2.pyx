# Written by Matthew Honnibal
# http://honnibal.wordpress.com
#
import random
from cymem.cymem cimport Pool

from libc.math cimport sqrt

cimport cython

cdef struct Point:
    double x
    double y

cdef class World:
    cdef Pool mem
    cdef int N
    cdef double* m
    cdef Point* r
    cdef Point* v
    cdef Point* F
    cdef readonly double dt
    def __init__(self, N, threads=1, m_min=1, m_max=30.0, r_max=50.0, v_max=4.0, dt=1e-3):
        self.mem = Pool()
        self.N = N
        self.m = <double*>self.mem.alloc(N, sizeof(double))
        self.r = <Point*>self.mem.alloc(N, sizeof(Point))
        self.v = <Point*>self.mem.alloc(N, sizeof(Point))
        self.F = <Point*>self.mem.alloc(N, sizeof(Point))
        for i in range(N):
            self.m[i] = random.uniform(m_min, m_max)
            self.r[i].x = random.uniform(-r_max, r_max)
            self.r[i].y = random.uniform(-r_max, r_max)
            self.v[i].x = random.uniform(-v_max, v_max)
            self.v[i].y = random.uniform(-v_max, v_max)
            self.F[i].x = 0
            self.F[i].y = 0
        self.dt = dt


@cython.cdivision(True)
def compute_F(World w):
    """Compute the force on each body in the world, w."""
    cdef int i, j
    cdef double s3, tmp
    cdef Point s
    cdef Point F
    for i in range(w.N):
        # Set all forces to zero. 
        w.F[i].x = 0
        w.F[i].y = 0
        for j in range(i+1, w.N):
            s.x = w.r[j].x - w.r[i].x
            s.y = w.r[j].y - w.r[i].y

            s3 = sqrt(s.x * s.x + s.y * s.y)
            s3 *= s3 * s3;

            tmp = w.m[i] * w.m[j] / s3
            F.x = tmp * s.x
            F.y = tmp * s.y

            w.F[i].x += F.x
            w.F[i].y += F.y

            w.F[j].x -= F.x
            w.F[j].y -= F.y


@cython.cdivision(True)
def evolve(World w, int steps):
    """Evolve the world, w, through the given number of steps."""
    cdef int _, i
    for _ in range(steps):
        compute_F(w)
        for i in range(w.N):
            w.v[i].x += w.F[i].x * w.dt / w.m[i]
            w.v[i].y += w.F[i].y * w.dt / w.m[i]
            w.r[i].x += w.v[i].x * w.dt
            w.r[i].y += w.v[i].y * w.dt
