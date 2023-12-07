import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import skimage
import config

def extract_wire(img):
    blurred = skimage.filters.gaussian(img, 6, preserve_range=True)
    diff = img - blurred
    mask = np.uint8(skimage.filters.apply_hysteresis_threshold(diff, 70, 120)) * 255
    mask = cv.dilate(mask, np.ones((3, 3)))
    skel = cv.ximgproc.thinning(mask)
    cc, rr = np.nonzero(skel.T)
    return rr[::10], cc[::10]

def pixel_to_ray(K, rr, cc):
    px_h = np.stack((cc, rr, np.ones(len(cc))), axis=-1)[..., None]
    rays = (np.linalg.inv(K) @ px_h).squeeze(-1)
    rays[..., 0] *= -1
    rays = rays / rays[..., 2:]
    return rays

def detect_full(imgs):
    rays = []
    for img in imgs:
        rr, cc = extract_wire(img)
        
        rays.append(pixel_to_ray(config.K, rr, cc))
    return rays
