## Estimate the Uncertainty For Triangulation
For most of the Bayesian Filters, the uncertainty plays a crucial role for estimation. However, it is difficult to compute the unceartinty of a triangulated landmark. Specifically, I consider the multiview feature triangulation where the prior camera poses with corresponding covariance matrices are available.

In this repo, I use unscented transform to estimate the covariance of a triangulated landmark. The simulator is borrowed from ['Semidefinite Relaxations for Robust Multiview Triangulation'](https://openaccess.thecvf.com/content/CVPR2023/html/Harenstam-Nielsen_Semidefinite_Relaxations_for_Robust_Multiview_Triangulation_CVPR_2023_paper.html). I use my own [visualization tool](https://github.com/HaktanM/cam_triangulation_visualization) to illustrate the triangulation steps. 

# What is Triangulation
