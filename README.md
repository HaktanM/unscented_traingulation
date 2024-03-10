## Estimate the Uncertainty For Triangulation
For most of the Bayesian Filters, the uncertainty plays a crucial role for estimation. However, it is difficult to compute the unceartinty of a triangulated landmark. Specifically, I consider the multiview feature triangulation where the prior camera poses with corresponding covariance matrices are available.

In this repo, I use unscented transform to estimate the covariance of a triangulated landmark. The simulator is borrowed from ['Semidefinite Relaxations for Robust Multiview Triangulation'](https://openaccess.thecvf.com/content/CVPR2023/html/Harenstam-Nielsen_Semidefinite_Relaxations_for_Robust_Multiview_Triangulation_CVPR_2023_paper.html). I use my own [visualization tool](https://github.com/HaktanM/cam_triangulation_visualization) to illustrate the triangulation steps. 

# What is Triangulation
Let us consider the ideal case first. You have captured the same scene from different angles. You exactly know the pose of the cameras. Assume that you have marked the pixel coordinates of a landmark within the images. Then, it is a simple geometry problem to compute the position of the landmark. This is called triangulation. You can examine the Figure below. 
![Ideal Triangulation](https://github.com/HaktanM/unscented_traingulation/blob/main/figures/ideal_triangulation.png)

# Add a Small Noise
The corner detector algorithms are not perfect. Hence, the pixel coordinates of the landmark is noisy. In that case, the rays from the center of the cameras to the image coordinates of the landmark do not intersect. Examine the next Figure.
![Noisy Observations](https://github.com/HaktanM/unscented_traingulation/blob/main/figures/noisy_triangulation.png)

In such a case, there might not be a feasible solution (Here, feasible solution means that the reprojection error is zero.). Rather, we employ an optimization algorithm to find the best candidate for the landmark position. There are different methods to compute the best candidate. I borrow the triangulation method from [OpenVINS](https://docs.openvins.com/update-featinit.html), a Bayesian SLAM algorithm. 

# Camera Poses are Corrupted
In terms of a SLAM or Structure from Motion algorithm, the camera poses is not known accuratly. Hence, the camera poses are also corrupted. I visualize the noisy camera poses with red. 
![Noisy Camera Poses](https://github.com/HaktanM/unscented_traingulation/blob/main/figures/cam_pose_noisy.png)

We can s
