import numpy as np

def sample_spherical_many(npoints=1, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def sample_spherical():
	return list(zip(*sample_spherical_many()))[0]