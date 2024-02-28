import numpy as np

class multiview_triangulation:
    def __init__(self,K):
        self.K = K
        self.K_inv = np.linalg.inv(K)
        
    def skew(self,vec):
        skew_sym = np.array(
            [
                0.0, -vec[2,0], vec[1,0],
                vec[2,0], 0.0, -vec[0,0],
                -vec[1,0], vec[0,0], 0.0
            ]
        ).reshape(3,3)
        
        return skew_sym
    
    def solve(self,poses,observations):
        R_A_to_G = poses.R[0]
        t_A_in_G = poses.t[0]

        R_G_to_A = R_A_to_G.transpose()
        t_G_in_A = - R_G_to_A @ t_A_in_G
        
        n_poses = observations[:,0].shape[0]

        A = np.zeros((3,3))
        b = np.zeros((3,1))
        for cam_idx in range(1,n_poses):
            R_C_to_G = poses.R[cam_idx]
            t_C_in_G = poses.t[cam_idx]
            
            R_C_to_A = R_G_to_A @ R_C_to_G
            
            t_C_in_A = R_G_to_A @ (t_C_in_G - t_A_in_G)
            
            ob = observations[cam_idx,:]
            ob_in_C = np.array([ob[0],ob[1],1.0]).reshape(3,1)
            ob_in_C_normalized = self.K_inv @  ob_in_C
            
            ob_in_A_normalized = R_C_to_A @ ob_in_C_normalized
            
            N_C_in_A = self.skew(ob_in_A_normalized)
            
            M = N_C_in_A.transpose() @ N_C_in_A
            v = M @ t_C_in_A
            
            A = A + M
            b = b + v.reshape(3,1)

        A_inv = np.linalg.inv(A)

        p_f_in_A = A_inv @ b
        p_f_in_A = p_f_in_A.reshape(3,1)
        
        p_f_in_G = R_A_to_G @ p_f_in_A + t_A_in_G.reshape(3,1)
        
        return p_f_in_G.reshape(3,1)