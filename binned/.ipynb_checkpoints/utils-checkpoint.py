import glob
import numpy as np
from scipy.stats import norm
import numba

def cross_matrix(x):
    r"""Calculate cross product matrix
    A[ij] = x_i * y_j - y_i * x_j
    """
    skv = np.roll(np.roll(np.diag(x.ravel()), 1, 1), -1, 0)
    return skv - skv.T
    
def rotate(ra1, dec1, ra2, dec2, ra3, dec3):
    r"""Rotation matrix for rotation of (ra1, dec1) onto (ra2, dec2).
    The rotation is performed on (ra3, dec3).
    """

    ra1 = np.atleast_1d(ra1)
    dec1 = np.atleast_1d(dec1)
    ra2 = np.atleast_1d(ra2)
    dec2 = np.atleast_1d(dec2)
    ra3 = np.atleast_1d(ra3)
    dec3 = np.atleast_1d(dec3)

    assert(
        len(ra1) == len(dec1) == len(ra2) == len(dec2) == len(ra3) == len(dec3)
    )

    cos_alpha = np.cos(ra2 - ra1) * np.cos(dec1) * np.cos(dec2) \
        + np.sin(dec1) * np.sin(dec2)

    # correct rounding errors
    cos_alpha[cos_alpha > 1] = 1
    cos_alpha[cos_alpha < -1] = -1

    alpha = np.arccos(cos_alpha)
    vec1 = np.vstack((np.cos(ra1) * np.cos(dec1),
                      np.sin(ra1) * np.cos(dec1),
                      np.sin(dec1))).T
    vec2 = np.vstack((np.cos(ra2) * np.cos(dec2),
                      np.sin(ra2) * np.cos(dec2),
                      np.sin(dec2))).T
    vec3 = np.vstack((np.cos(ra3) * np.cos(dec3),
                      np.sin(ra3) * np.cos(dec3),
                      np.sin(dec3))).T
    nvec = np.cross(vec1, vec2)
    norm = np.sqrt(np.sum(nvec**2, axis=1))
    nvec[norm > 0] /= norm[np.newaxis, norm > 0].T

    one = np.diag(np.ones(3, dtype=np.float32))
    nTn = np.array([np.outer(nv, nv) for nv in nvec])
    nx = np.array([cross_matrix(nv) for nv in nvec])

    R = np.array([(1. - np.cos(a)) * nTn_i + np.cos(a) * one + np.sin(a) * nx_i
                  for a, nTn_i, nx_i in zip(alpha, nTn, nx)])
    vec = np.array([np.dot(R_i, vec_i.T) for R_i, vec_i in zip(R, vec3)])

    ra = np.arctan2(vec[:, 1], vec[:, 0])
    dec = np.arcsin(vec[:, 2])

    ra += np.where(ra < 0., 2. * np.pi, 0.)

    return ra, dec


def prepare_simulation(source_ra, source_dec, # degrees
                       window = 5, # degrees
                       E0 = 1000, # GeV,
                       gamma = 2,
                       time_mean = 57000,
                       time_sigma = 100,
                       simfile = "/data/mjlarson/datasets/version-003-p02/IC86_2012_MC.npy"):
    # Load the simulation
    sim = np.load(simfile)
    
    source_ra, source_dec, window = np.radians((source_ra,
                                                source_dec,
                                                window))

    # Only select events within 5 degrees in dec
    mask = np.abs(sim['trueDec']-source_dec) < window
    sim = sim[mask]
    
    # Rotate all of the events to be at truth. This will make life easier
    # for the students.
    source_ra, source_dec = (np.ones(len(sim['trueRa']))*source_ra,
                             np.ones(len(sim['trueRa']))*source_dec)
    newra, newdec = rotate(sim['trueRa'], sim['trueDec'], 
                           source_ra, source_dec,
                           sim['ra'], sim['dec'])
    weight = sim['ow'] * (sim['trueE']/E0)**-gamma
    time = norm.rvs(time_mean, time_sigma, len(weight))
    
    # Convert the weight to a probability per event.
    # This lets us give answers in number of events instead of flux normalization units
    probability_weight = weight/weight.sum()
    
    output_dtype = [("ra", float), ("dec", float), ('angErr', float), 
                    ("logE", float), ("time", float), 
                    ("probability_weight", float), ("weight", float)]
    
    return np.array(list(zip(newra, newdec, sim['angErr'], sim['logE'], time, 
                             probability_weight, weight)), dtype=output_dtype)
    
    
    def prepare_data(datafiles = glob.glob("/data/mjlarson/datasets/version-003-p02/IC86_201[2-9]_exp.npy")):
    output = []
    for f in sorted(datafiles):
        output.append(np.load(f))
        
    # We're going to randomize the times, since the times from this dataset aren't unblinded
    output = np.concatenate(output)
    output['time'] = np.random.shuffle(output['time'])
        
    return output

def interp_hist(x, hist, bins):
    centers = bins[:-1]+np.diff(bins)
    return np.interp(x, centers, hist)
    
    
def angular_distance(src_ra, src_dec, ra, dec):
    r""" Compute angular distance between source and location """

    sinDec = np.sin(dec)

    cos_dec = np.sqrt(1. - sinDec**2)

    cosDist = (
        np.cos(src_ra - ra) * np.cos(src_dec) * cos_dec +
        np.sin(src_dec) * sinDec
    )

    # handle possible floating precision errors
    cosDist[np.isclose(cosDist, -1.) & (cosDist < -1)] = -1.
    cosDist[np.isclose(cosDist, 1.) & (cosDist > 1)] = 1.
    return np.arccos(cosDist)
