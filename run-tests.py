
import os
import time

import numpy as np

import lib
import lib.sim2

def test_fn(evolve_fn, name, steps=1000, dt=1e-3, bodies=101, threads=1):
    print "\n"
    if name == 'Cython':
        w = lib.sim2.World(bodies, threads=threads, dt=dt)
    else:
        w = lib.World(bodies, threads=threads, dt=dt)
    # Test the speed of the evolution function. 
    
    t0 = time.time()
    evolve_fn(w, steps)
    t1 = time.time()
    print "{0} ({1}): {2} steps/sec".format(
        name, threads, int(steps / (t1 - t0)))
    
    

if __name__ == "__main__":
    # Single CPU only tests. 
    test_fn(lib.sim2.evolve, "Cython", steps=32000)
    
    test_fn(lib.evolve_c_simple1, "C Simple 1", steps=32000)
    test_fn(lib.evolve_c_simple2, "C Simple 2", steps=32000)
    
