from simulator import triangulation
import numpy as np

from visualizer.vis_class import _vis

#np.random.seed(2)
def skew(vec):
    skew_sym = np.array(
        [
            0.0, -vec[2,0], vec[1,0],
            vec[2,0], 0.0, -vec[0,0],
            -vec[1,0], vec[0,0], 0.0
        ]
    ).reshape(3,3)
    
    return skew_sym

n_poses = 100
K = np.array([
        [640, 0, 300],
        [0., 512, 250],
        [0., 0., 1.],
    ]).reshape(3,3)
K_inv = np.linalg.inv(K)


radius = 2500
observation_noise=20.0
problem = triangulation.get_aholt_problem(n_poses, mode="sphere", K=K, radius=radius)
observations = problem.get_observations(sigma=observation_noise)


R_A_to_G = problem.poses.R[0]
t_A_in_G = problem.poses.t[0]

R_G_to_A = R_A_to_G.transpose()
t_G_in_A = - R_G_to_A @ t_A_in_G

A = np.zeros((3,3))
b = np.zeros((3,1))
for cam_idx in range(1,n_poses):
    R_C_to_G = problem.poses.R[cam_idx]
    t_C_in_G = problem.poses.t[cam_idx]
     
    R_C_to_A = R_G_to_A @ R_C_to_G
    
    t_C_in_A = R_G_to_A @ (t_C_in_G - t_A_in_G)
    
    ob = observations[cam_idx,:]
    ob_in_C = np.array([ob[0],ob[1],1.0]).reshape(3,1)
    ob_in_C_normalized = K_inv @  ob_in_C
    
    print(f"ob_in_C\n{ob_in_C}")
    print(f"ob_in_C_normalized\n{ob_in_C_normalized}")
    ob_in_A_normalized = R_C_to_A @ ob_in_C_normalized
    
    N_C_in_A = skew(ob_in_A_normalized)
    
    M = N_C_in_A.transpose() @ N_C_in_A
    v = M @ t_C_in_A
    
    A = A + M
    b = b + v.reshape(3,1)

A_inv = np.linalg.inv(A)

p_f_in_A = A_inv @ b
p_f_in_A = p_f_in_A.reshape(3,1)

p_f_in_G = R_A_to_G @ p_f_in_A + t_A_in_G.reshape(3,1)

print(f"gt : \n{problem.point}")
print(f"es : \n{p_f_in_G}")
