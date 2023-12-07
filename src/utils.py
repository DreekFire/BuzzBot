import numpy as np

def skew(vecs):
    zeros = np.zeros(vecs.shape[:-1])
    arr = np.array([[        zeros, -vecs[..., 2],  vecs[..., 1]],
                    [ vecs[..., 2],         zeros, -vecs[..., 0]],
                    [-vecs[..., 1],  vecs[..., 0],         zeros]])
    # (3, 3, *batch)
    return np.moveaxis(arr, (0, 1), (-2, -1))

def to_homogeneous(pts):
    return np.concatenate((pts, np.ones((len(pts), 1))), axis=1)

def project(K, extrinsic, pts):
    P = K @ extrinsic
    local_pts = (P @ pts[..., None])[..., 0]
    return local_pts / local_pts[..., 2:]