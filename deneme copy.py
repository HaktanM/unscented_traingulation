from scipy.spatial.transform import Rotation as ori
from simulator import triangulation
import numpy as np
from simulator.se3 import Se3

from visualizer.vis_class import _vis

from solver.solver import multiview_triangulation
#np.random.seed(2)

n_poses = 3
K = np.array([
        [640, 0, 300],
        [0., 512, 250],
        [0., 0., 1.],
    ]).reshape(3,3)

radius = 10
observation_noise=10.0
problem = triangulation.get_aholt_problem(n_poses, mode="sphere", K=K, radius=radius)
observations = problem.get_observations(sigma=observation_noise)

solver = multiview_triangulation(K)

estimated = solver.solve(problem.poses,observations)


C_pos = np.array([
    50.0**2, 20.0**2, 15.0**2,
    20.0**2, 80.0**2, 25.0**2,
    15.0**2, 25.0**2, 75.0**2
]).reshape(3,3) # meter square

C_ori_deg = np.array([
    0.200**2, 0.018**2, 0.001**2,
    0.018**2, 0.150**2, 0.020**2,
    0.001**2, 0.020**2, 0.250**2
]).reshape(3,3) # degree square

C_pos = C_pos / 100**2
C_ori_deg = C_ori_deg*0.1

error_mean = np.zeros(3)

t_noisy = np.zeros((n_poses,3))
R_noisy = np.zeros((n_poses,3,3))
for cam_idx in range(n_poses):
    t = problem.poses.t[cam_idx].reshape(1,3)
    t_noise = np.random.multivariate_normal(error_mean,C_pos).reshape(1,3)
    t_noisy[cam_idx,:] = t + t_noise
    
    R = problem.poses.R[cam_idx].reshape(3,3)
    euler_noise = np.random.multivariate_normal(error_mean,C_ori_deg)
    R_noise = ori.from_euler("xyz", euler_noise, degrees=True).as_matrix()
    R_noisy[cam_idx,:,:] = R_noise @ R
    
noisy_poses = Se3(R_noisy,t_noisy)

solver = multiview_triangulation(K)

estimated_noisy = solver.solve(noisy_poses,observations)
estimated_accurate = solver.solve(problem.poses,observations)

print(f"Original Point : \n{problem.point}")
print(f"estimated_noisy : \n{estimated_noisy}")
print(f"estimated_accurate : \n{estimated_accurate}")


vis = _vis(K,radius*1.01)

for cam_idx in range(n_poses):
    vis.add_cam(problem.poses.t[cam_idx],problem.poses.R[cam_idx])
    vis.add_cam_noisy(noisy_poses.t[cam_idx],noisy_poses.R[cam_idx])
    
vis.plot_noisy_observations(observations)
vis.plot_noisy_observations_on_noisy_cams(observations)

vis.ax.scatter3D(problem.point[0], problem.point[1], problem.point[2], color='green', s = 50)
vis.ax.scatter3D(estimated_accurate[0,0], estimated_accurate[1,0], estimated_accurate[2,0], color='black', s = 50)
vis.ax.scatter3D(estimated_noisy[0,0], estimated_noisy[1,0], estimated_noisy[2,0], color='orange', s = 50)

vis.get_plot()