import numpy as np

from simulator import so3, se3, geometry
from simulator.se3 import Se3


class TriangulationProblem:
    def __init__(self, poses: Se3, point: np.array, K: np.array):
        self.poses = poses
        self.n_poses = len(poses)
        self.point = point
        self.K = K
        self.K_inv = np.linalg.inv(K)

    def get_observations(self, sigma=0.0):
        x = geometry.reproject(self.point, self.poses, self.K)
        x += np.random.randn(*x.shape) * sigma
        return x

    def get_transformed(self):
        return geometry.transform(self.poses.inverse(), self.point, self.K_inv)

    @property
    def F(self):
        return geometry.get_fundamental(se3.get_relative(self.poses), self.K_inv)

    def __str__(self):
        return f"Triangulation problem\n" \
               f"n_poses: {len(self.poses)}\n" \
               f"point: {self.point}\n" \
               f"K: {self.K}"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        return TriangulationProblem(
            poses=self.poses[item],
            point=self.point,
            K=self.K,
        )
        
def get_aholt_problem(n_poses, mode="sphere", angle_jitter_deg=0., K=None, radius=2.):
    """Sphere experiment from Aholt et al"""

    def get_problem():
        if mode == "sphere":
            t = np.random.randn(n_poses, 3) - 0.5
        elif mode == "circle":
            t = np.zeros((n_poses, 3))
            t[:, :2] = np.random.randn(n_poses, 2)
        t *= radius / np.linalg.norm(t, axis=1)[:, None]
        R = so3.get_rotaitons_facing_point(np.zeros(3), t)
        if angle_jitter_deg > 0:
            R = np.einsum('...ij, ...jk->...ik', R,
                          so3.rotvec_to_r(np.random.randn(n_poses, 3) * np.pi * angle_jitter_deg / 180))
        return TriangulationProblem(
            poses=Se3(R, t),
            point=np.random.rand(3),
            K=K if K is not None else np.eye(3)
        )

    problem = get_problem()

    return problem