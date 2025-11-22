import numpy as np
import torch
import math

class PCDRandomRotate(object):
    """
    Randomly rotate the point cloud around configurable axes.

    Args:
        angle_range: scalar in degrees. Each enabled axis samples its angle uniformly
            from [-angle_range, +angle_range]. (Backward compat default 180.)
        axes: subset of {'x','y','z'}. 'y' (default) replicates the previous
            Y-only behaviour. 'xy' rotates around X and Y simultaneously
            (R = Ry @ Rx). 'xyz' similarly composes all three.
    """
    def __init__(self, angle_range=180, axes='y'):
        self.angle_range = float(angle_range)
        self.axes = ''.join(a.lower() for a in axes if a.lower() in 'xyz')
        if not self.axes:
            self.axes = 'y'

    @staticmethod
    def _rot_matrix(axis, angle_deg):
        a = math.radians(angle_deg)
        c, s = math.cos(a), math.sin(a)
        if axis == 'x':
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
        if axis == 'y':
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        # z
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

    def __call__(self, pcd):
        """
        Args:
            pcd: (N, 3) torch.Tensor or numpy.ndarray
        """
        # Compose rotations in axis order ('xy' → Ry @ Rx, 'xyz' → Rz @ Ry @ Rx)
        R = np.eye(3, dtype=np.float32)
        for axis in self.axes:
            angle = np.random.uniform(-self.angle_range, self.angle_range)
            R = self._rot_matrix(axis, angle) @ R

        if isinstance(pcd, torch.Tensor):
            R_t = torch.from_numpy(R).to(pcd.device)
            return torch.matmul(pcd, R_t.t())
        return np.dot(pcd, R.T)

class PCDRandomJitter(object):
    """
    Add random Gaussian noise to point coordinates.
    """
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, pcd):
        if isinstance(pcd, torch.Tensor):
            noise = torch.randn_like(pcd) * self.sigma
            noise = torch.clamp(noise, -self.clip, self.clip)
            return pcd + noise
        else:
            noise = np.random.normal(0, self.sigma, pcd.shape)
            noise = np.clip(noise, -self.clip, self.clip)
            return pcd + noise.astype(pcd.dtype)

class PCDRandomScale(object):
    """
    Randomly scale the point cloud uniformly.
    """
    def __init__(self, scale_low=0.8, scale_high=1.2):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pcd):
        scale = np.random.uniform(self.scale_low, self.scale_high)
        return pcd * scale

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pcd):
        for t in self.transforms:
            pcd = t(pcd)
        return pcd
