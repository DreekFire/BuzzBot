import numpy as np
import scipy as sp
import config
from utils import *
from matplotlib import pyplot as plt

# Every function in this file expects 2D homogeneous coordinates

# Arguments:
# epipolar_lines: E @ x
# points_h_2d: x'
# Returns:
# intersections: points where the epipolar lines intersect the curve
# n1: point idx
def intersect_curves(epipolar_lines, pts):
    # corresponding points satisfy x'Ex = 0
    # so find when x'Ex changes signs
    # TODO: maybe include points that get very close to zero and are closer than their neighbors
    offsets = np.einsum('ik,jk->ij', epipolar_lines, pts) #np.sum(epipolar_lines[:, None] * pts[None], axis=-1) # (n1, 1, 3) * (1, n2, 3) -> (n1, n2)
    sgn = np.sign(offsets)
    flip = sgn[..., 1:] - sgn[..., :-1]
    # n1: index of point in first camera view
    # n2: index of point in second camera view
    n1, n2 = np.nonzero(flip)
    diff = pts[n2] - pts[n2 + 1]
    rate = np.sum(epipolar_lines[n1] * diff, axis=-1)
    factor = (-offsets[n1, n2] / rate)[..., None]
    rate_normalized = rate / (np.linalg.norm(epipolar_lines[n1], axis=-1) * np.linalg.norm(diff, axis=-1))
    return pts[n2] + factor * diff, rate_normalized, n1, offsets

def get_depths(x_matched, intersections, R, t):
    # get ray directions of matching points
    global_pts2 = (R @ intersections[..., None])[..., 0]
    # find perpendicular displacement from origin to each ray
    # by subtracting projection of translation onto ray
    projection = (np.sum(t * global_pts2, axis=-1, keepdims=True)
                                   / np.sum(global_pts2 * global_pts2, axis=-1, keepdims=True)) * global_pts2
    perpendicular = t - projection
    # find depths as length of perpendiculars divided by component of first camera rays on perpendiculars
    # basically, rays cast from camera 1 origin should reach the ray cast from camera 2
    return np.sum(perpendicular * perpendicular, axis=-1) / np.sum(x_matched * perpendicular, axis=-1)

def match_candidates(x, depths_1, depths_2):
    err = [c1[..., None] - c2[..., None, :] for c1, c2 in zip(depths_1, depths_2)]
    assert len(err) == len(x)

    depths = []
    last_point = None
    for i, er in enumerate(err):
        if er.size == 0:
            last_point = None
            depths.append(-1)
            print('no matches found')
            continue
        penalty = None
        if last_point is not None:
            new_pts_1 = x[i, None] * depths_1[i][..., None]
            new_pts_2 = x[i, None] * depths_2[i][..., None]
            diff_1 = np.linalg.norm(new_pts_1 - last_point[None], axis=-1) ** 2
            diff_2 = np.linalg.norm(new_pts_2 - last_point[None], axis=-1) ** 2
            penalty = diff_1[..., None] + diff_2[..., None, :]
            print("applying penalty: ", penalty)
            assert penalty.shape == er.shape, penalty.shape
            er = er + 2 * penalty
        acc = np.argmin(np.abs(er))
        idx = np.unravel_index(acc, er.shape)
        d1 = depths_1[i][idx[0]]
        d2 = depths_2[i][idx[1]]
        if np.abs(d2 - d1) < 0.03:
            d = 0.5 * (d1 + d2)
            depths.append(d)
            if penalty is None or penalty[idx[0], idx[1]] < 0.02:
                last_point = x[i] * d
        else:
            last_point = None
            depths.append(-2)
            print('error too large')
    depths = np.array(depths)

    return depths, depths >= 0

# def animate_epipolar(epipolar_lines, x_1, x_2, x_3, intersections_1, intersections_2, n1, n2, test_pt):
#     fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

#     axs[1, 0].plot(*x_1[..., :2].T, color='red')
#     axs[1, 1].plot(*x_2[..., :2].T, color='red')
#     axs[0, 0].plot(*x_3[..., :2].T, color='red')
#     axs[1, 0].scatter(*x_1[test_pt, :2], color='blue')
#     axs[1, 1].scatter(*intersections_1[n1==test_pt][..., :2].T, color='blue')
#     axs[0, 0].scatter(*intersections_2[n2==test_pt][..., :2].T, color='blue')

#     def points_on_epipolar(epipolar):
#         a, b, c = epipolar
#         p1 = np.array([a, b])
#         p1 = -p1 * c / np.sum(p1 * p1)
#         p2 = p1 + np.array([b, -a])
#         return p1, p2

#     p1, p2 = points_on_epipolar(epipolar_lines[0][test_pt])
#     axs[1, 1].axline(p1, p2, color='green')

#     p1, p2 = points_on_epipolar(epipolar_lines[1][test_pt])
#     axs[0, 0].axline(p1, p2, color='green')
#     plt.show()

# def points_on_epipolar(epipolar):
#     a, b, c = epipolar
#     p1 = np.array([a, b])
#     p1 = -p1 * c / np.sum(p1 * p1)
#     p2 = p1 + np.array([b, -a])
#     return p1, p2

def points_on_epipolar(epipolar):
    a, b, c = epipolar
    # p1 = np.array([a, b])
    # p1 = -p1 * c / np.sum(p1 * p1)
    if np.abs(b) < 1e-7:
        return np.array([(b - c) / a, -1]), np.array([(-b - c) / a, 1])
    return np.array([-1, (a - c) / b]), np.array([1, (-a - c) / b])

def visualize_epipolar(epipolar_lines, x_1, x_2, x_3, off, images=None):
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    axs[1, 0].plot(*x_1[..., :2].T, color='red')
    axs[1, 1].plot(*x_2[..., :2].T, color='red')
    axs[0, 0].plot(*x_3[..., :2].T, color='red')
    scatter_0 = axs[1, 0].scatter(*x_1[0, :2], color='blue')
    # scatter_1 = axs[1, 1].scatter(*intersections_1[n1==0][..., :2].T, color='blue')
    # scatter_2 = axs[0, 0].scatter(*intersections_2[n2==0][..., :2].T, color='blue')
    scatter_1 = axs[1, 1].scatter(*x_2[:, :2].T, s=3, color='blue')
    scatter_2 = axs[0, 0].scatter(*x_3[:, :2].T, s=3, color='blue')

    # points = x_1[None, TEST_PT] * np.linspace(0, 3, 50)[..., None]
    # assert points.shape == (50, 3)
    # points = np.concatenate((points, np.ones((50, 1))), axis=-1)
    # assert points.shape == (50, 4)
    # print(points)
    # points = (extrin[0] @ points[..., None]).squeeze(-1).T[:2]
    # print(points)
    # breakpoint()
    # axs[1, 1].scatter(*points, s=4, color='red')
    
    p1, p2 = points_on_epipolar(epipolar_lines[0][0])
    ax1 = axs[1, 1].plot(*zip(p1, p2), color='green')
    # ax1 = axs[1, 1].plot(*coords.T, color='green')

    p1, p2 = points_on_epipolar(epipolar_lines[1][0])
    ax2 = axs[0, 0].plot(*zip(p1, p2), color='green')
    # ax2 = axs[0, 0].plot(*coords.T, color='green')

    corners = np.array([[0, 0, 1],
                        [config.W, config.H, 1]]).T
    corners = np.linalg.inv(config.K) @ corners
    extents = corners[:2].flatten()
    axs[0, 0].set_xlim(extents[:2])
    axs[0, 0].set_ylim(extents[2:])
    if images:
        axs[0, 0].imshow(images[2][::-1, ::-1], extent=extents)
        axs[1, 0].imshow(images[0][::-1, ::-1], extent=extents)
        axs[1, 1].imshow(images[1][::-1, ::-1], extent=extents)
    
    def animate(i):
        scatter_0.set_offsets(x_1[i, :2])
        # scatter_1.set_offsets(intersections_1[n1==i][..., :2])
        # scatter_2.set_offsets(intersections_2[n2==i][..., :2])
        scatter_1.set_color(['red' if ofs > 0 else 'blue' for ofs in off[0][i]])
        scatter_2.set_color(['red' if ofs > 0 else 'blue' for ofs in off[1][i]])

        p1, p2 = points_on_epipolar(epipolar_lines[0][i])
        ax1[0].set_data(*zip(p1, p2))

        p1, p2 = points_on_epipolar(epipolar_lines[1][i])
        ax2[0].set_data(*zip(p1, p2))

        return [scatter_0, scatter_1, scatter_2, ax1[0], ax2[0]]

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, animate,
                        frames = len(x_1), interval = 200, blit = True) 
    anim.save('intersections.gif')
    plt.show()

def reconstruct_full(x, x_prime, rot, pos, essential):
    intersections = []
    n = []
    depths = []
    epipolar = []
    offsets = []
    for i in range(2):
        epipolar_lines = (essential[i] @ x[..., None])[..., 0]
        epipolar.append(epipolar_lines)
        assert epipolar_lines.shape == (len(x), 3)
        inter, rate_normalized, nn, off = intersect_curves(epipolar_lines, x_prime[i])
        offsets.append(off)
        d = get_depths(x[nn], inter, rot[i], pos[i])
        plus = d > 0
        d_list = [[] for _ in range(len(x))]
        nn = nn[plus]
        d = d[plus]
        inter = inter[plus]
        n.append(nn)
        intersections.append(inter)
        for j, idx in enumerate(nn):
            d_list[idx].append(d[j])
        d_list = [np.array(lst) for lst in d_list]
        depths.append(d_list)

    visualize_epipolar(epipolar, x, *x_prime, offsets)

    selected_depths, mask = match_candidates(x, *depths)
    return x * selected_depths[..., None], mask