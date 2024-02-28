from scipy.spatial.transform import Rotation as ori
from simulator import triangulation
import numpy as np
from simulator.se3 import Se3

from visualizer.vis_class import _vis

from solver.solver import multiview_triangulation
# np.random.seed(2)

n_poses = 5
K = np.array([
        [640, 0, 300],
        [0., 512, 250],
        [0., 0., 1.],
    ]).reshape(3,3)

radius = 10
observation_noise=0.0
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
    0.300**2, 0.018**2, 0.001**2,
    0.018**2, 0.150**2, 0.020**2,
    0.001**2, 0.020**2, 0.250**2
]).reshape(3,3) # degree square

C_pos = C_pos / 100**2
C_ori_deg = C_ori_deg*0.1

error_mean = np.zeros(3)

SigmaPoints = []

# The very first sigma point is the mean
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
    
noisy_poses_mean = Se3(R_noisy,t_noisy)
SigmaPoints.append(noisy_poses_mean)

chol_C_pos = np.linalg.cholesky(C_pos).reshape(3,3)
chol_C_ori_deg = np.linalg.cholesky(C_ori_deg).reshape(3,3) 

for cam_idx in range(1,n_poses):
    t_mean = np.copy(noisy_poses_mean.t[cam_idx].reshape(3,1))
    R_mean = np.copy(noisy_poses_mean.R[cam_idx].reshape(3,3))
    
    for par_idx in range(3):       
        t_sigma_vec = chol_C_pos[:,par_idx].reshape(-1)
        R_sigma_vec = chol_C_ori_deg[:,par_idx]
        R_sigma_mat = ori.from_euler("xyz", R_sigma_vec, degrees=True).as_matrix()
        R_sigma_mat_neg = ori.from_euler("xyz", -R_sigma_vec, degrees=True).as_matrix()
        
        t_noisy_temp = np.copy(t_noisy)
        t_noisy_temp[cam_idx,:] = t_noisy_temp[cam_idx,:] + t_sigma_vec
        sigmaPoint = Se3(R_noisy,t_noisy_temp)
        SigmaPoints.append(sigmaPoint)

        t_noisy_temp = np.copy(t_noisy)
        t_noisy_temp[cam_idx,:] = t_noisy_temp[cam_idx,:] - t_sigma_vec
        sigmaPoint = Se3(R_noisy,t_noisy_temp)
        SigmaPoints.append(sigmaPoint)

        R_noisy_temp = np.copy(R_noisy)
        R_noisy_temp[cam_idx,:,:] = (R_noisy_temp[cam_idx,:,:].reshape(3,3)) @ R_sigma_mat
        sigmaPoint = Se3(R_noisy_temp,t_noisy)
        SigmaPoints.append(sigmaPoint)
        
        R_noisy_temp = np.copy(R_noisy)
        R_noisy_temp[cam_idx,:,:] = (R_noisy_temp[cam_idx,:,:].reshape(3,3)) @ R_sigma_mat_neg
        sigmaPoint = Se3(R_noisy_temp,t_noisy)
        SigmaPoints.append(sigmaPoint)
        
        

mapped_sigma_points = []
solver = multiview_triangulation(K)

pi_0 = -100
Weights = []
Weights.append(pi_0)
for sigma_idx in range(1,len(SigmaPoints)):
    w = (1 - pi_0) / (len(SigmaPoints)-1)
    Weights.append(w)

estimation_mean = np.zeros((3,1))
for sigmaPoint,w in zip(SigmaPoints,Weights):
    estimation = solver.solve(sigmaPoint,observations)
    mapped_sigma_points.append(estimation)
    
    estimation_mean += w * estimation

estimation_mean = estimation_mean.reshape(3,1)
estimation_cov = np.zeros((3,3))
for estimation,w in zip(mapped_sigma_points,Weights):
    diff = (estimation - estimation_mean).reshape(3,1)
    estimation_cov = estimation_cov + w * diff @ diff.transpose()
    
estimated_accurate = solver.solve(problem.poses,observations)



vis = _vis(K,radius*1.2)

for cam_idx in range(n_poses):
    vis.add_cam(problem.poses.t[cam_idx],problem.poses.R[cam_idx])
    vis.add_cam_noisy(noisy_poses_mean.t[cam_idx],noisy_poses_mean.R[cam_idx])
    
vis.plot_noisy_observations(observations)
vis.plot_noisy_observations_on_noisy_cams(observations)

vis.ax.scatter3D(problem.point[0], problem.point[1], problem.point[2], color='green', s = 50)
vis.ax.scatter3D(estimated_accurate[0,0], estimated_accurate[1,0], estimated_accurate[2,0], color='black', s = 50)

first = True
for item in mapped_sigma_points:
    if first:
        first = False
    else:    
        vis.ax.scatter3D(item[0,0], item[1,0], item[2,0], color='orange', s = 20)
    
vis.ax.scatter3D(mapped_sigma_points[0][0,0], mapped_sigma_points[0][1,0], mapped_sigma_points[0][2,0], color='red', s = 50)
vis.ax.scatter3D(estimation_mean[0,0], estimation_mean[1,0], estimation_mean[2,0], color='blue', s = 50)

error = (estimation_mean.reshape(3,1) - problem.point.reshape(3,1)).reshape(3,1)

print(estimation_cov)
gating = error.reshape(1,3) @ np.linalg.inv(estimation_cov) @ error
print(gating)

vis.get_plot()