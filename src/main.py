import reconstruction, detection
import config
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from utils import *

image_files = [
    'new_sawyer_test/0.png',
    'new_sawyer_test/1.png',
    'new_sawyer_test/2.png',
]

def visualize_detection(imgs, pts):
    pixels = []
    for i in range(len(pts)):
        px = (config.K @ pts[i][..., None])[..., 0]
        px = (px[:, :2] / px[:, None, 2])
        pixels.append(px)
    fig, axs = plt.subplots(1, len(imgs))
    for i in range(len(imgs)):
        axs[i].imshow(imgs[i][:, ::-1], cmap='gray')
        axs[i].scatter(*pixels[i].T, s=2, c=np.arange(len(pixels[i])), cmap='viridis')
    
    plt.savefig('detection.png')
    plt.show()

def visualize(imgs, points_3d, points_2d, extrinsic):
    fig, axs = plt.subplots(1, len(extrinsic))
    for i, ext in enumerate(extrinsic):
        axs[i].imshow(imgs[i][:, ::-1], cmap='gray')
        scale = 1 # imgs[i].shape[0] / config.W
        projected = project(config.K, extrinsic[i], points_3d)
        projected = projected[..., :2] / projected[..., 2:]
        pts = (np.linalg.inv(config.K) @ points_2d[i][..., None]).squeeze(-1)
        pts = (pts[:, :2] / pts[:, 2:])
        axs[i].scatter(*projected.T, s=6, c=np.arange(len(projected)), cmap='viridis')
        axs[i].scatter(*pts.T, s=6, c=np.arange(len(pts)), cmap='inferno')

    plt.savefig('final.png')
    plt.show()

def get_points(images, transformations):
    G = np.linalg.inv(transformations[0]) @ transformations
    # G = config.G

    pos = G[..., :3, 3]
    rot = G[..., :3, :3]
    
    R = G[:, :3, :3].swapaxes(-1, -2)
    T = (-R @ G[:, :3, 3:]).squeeze(-1)
    essential = skew(T) @ R

    # detect
    pts = detection.detect_full(images)
    # reconstruct
    rays, mask = reconstruction.reconstruct_full(pts[0], pts[1:], rot[1:], pos[1:], essential[1:])
    points_3d = (G[0] @ np.concatenate((rays, np.ones((len(rays), 1))), axis=-1)[..., None])[mask].squeeze(-1)
    diff = np.linalg.norm(np.diff(points_3d, axis=0), axis=-1)
    diff = np.concatenate(([np.inf], diff, [np.inf]))
    new_mask = np.minimum(diff[1:], diff[:-1]) < 0.1
    points_3d = points_3d[new_mask]
    visualize(images, points_3d, pts, np.linalg.inv(G)[:, :3])
    return points_3d

def smooth_chord(points, idx):
    tail = points[max(idx, 0)]
    diff = points[idx] - tail
    return diff / np.linalg.norm(diff)

def main():
    images = [cv.imread(img_file, cv.IMREAD_GRAYSCALE) for img_file in image_files]
    images = [cv.resize(img, (config.W, config.H))[:, ::-1] for img in images]
    
    pts = get_points(images, config.G)
    

if __name__ == "__main__":
    main()