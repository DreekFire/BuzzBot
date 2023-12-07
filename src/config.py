import numpy as np
from scipy.spatial.transform import Rotation

# height: 480
# width: 752
# distortion_model: "plumb_bob"
D = [-0.442, 0.275, 0.001, 0.001, -0.13]

W = 752
H = 480

K = [[624.150024,        0.0, 374.570007,],
     [       0.0, 623.059998, 233.796997,],
     [       0.0,        0.0,        1.0,]]
K = np.array(K)

# binning_x: 1
# binning_y: 1
# roi:
#   x_offset: 0
#   y_offset: 0
#   height: 480
#   width: 752
#   do_rectify: False

# T = np.array([
#     [0.25,   0, 0],
#     [1.0,   0, 0],
#     [0.5, 0.5, 0]
# ])

T = np.array([
     [0.715, -0.246, 0.358],
     [0.726,  0.464, 0.371],
     [0.681,  0.121, 0.426],
])


R_quaternion = np.array([
     [-0.667, 0.702, 0.124, 0.216],
     [0.700, -0.653, 0.240, 0.162],
     [-0.706, 0.708, 0.000, 0.003],
])

# T = np.array(
#     [[0.666,  0.428, 0.378],
#      [0.821, -0.344, 0.297],
#      [0.857, -0.067, 0.562]]
# )

# T = np.array(
#     [[-0.428, 0.666, 0.378],
#      [ 0.344, 0.821, 0.297],
#      [ 0.067, 0.857, 0.562]]
# )

# R_quaternion = [[0.694, 0.678, -0.139,  0.195],
# 	           [0.694, 0.658,  0.163, -0.243],
# 	           [0.658, 0.752,  0.025, -0.039]]

R = Rotation.from_quat(R_quaternion).as_matrix()

# R = np.eye(3)[None].repeat(3, 0)

G = np.zeros((3, 4, 4))
G[:, :3, :3] = R
G[:, :3, 3] = T
G[:, 3, 3] = 1
