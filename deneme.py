from simulator import triangulation
import numpy as np

from visualizer.vis_class import _vis

from solver.solver import multiview_triangulation
#np.random.seed(2)

n_poses = 3
K = np.array([
        [640, 0, 300],
        [0., 512, 250],
        [0., 0., 1.],
    ]).reshape(3,3)

radius = 2000
observation_noise=10.0
problem = triangulation.get_aholt_problem(n_poses, mode="sphere", K=K, radius=radius)
observations = problem.get_observations(sigma=observation_noise)

solver = multiview_triangulation(K)

estimated = solver.solve(problem.poses,observations)

print(f"Original Point : \n{problem.point}")
print(f"Estimated Point : \n{estimated}")

vis = _vis(K,radius*1.01)

for cam_idx in range(n_poses):
    vis.add_cam(problem.poses.t[cam_idx],problem.poses.R[cam_idx])
vis.add_landmark(problem.point)

vis.plot_noisy_observations(observations)

vis.ax.scatter3D(estimated[0,0], estimated[1,0], estimated[2,0], marker=(5, 2), color='red', s = 50)

vis.get_plot()

# print(problem.point)