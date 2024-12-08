import os
import imageio
import trimesh
import numpy as np


def normalize_pc(v):
    vmax, vmin = v.max(axis=0), v.min(axis=0)
    diag = vmax - vmin
    c = v.mean(axis=0)
    norm = 1 / np.linalg.norm(diag)
    return np.array(c), np.array(norm)


def depth_to_pc(depth_map, intrinsic_matrix, mask):
    h, w = depth_map.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    # Create a mesh grid of image coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u, v = u.flatten(), v.flatten()

    # Flatten the depth map and color image
    z = depth_map.flatten()
    # Compute X, Y, Z coordinates in 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack((x, y, z), axis=-1)

    # Filter out points where depth is zero and keep valid indices
    valid_mask = (z > 0) & (mask.flatten() > 0)
    valid_points = points[valid_mask]
    
    return valid_points


def farthest_point_sampling(points, num_samples):
    points = np.array(points)  # Ensure points are a NumPy array
    N, D = points.shape

    if num_samples > N:
        raise ValueError("Number of samples cannot exceed the number of available points.")
    sampled_indices = np.zeros(num_samples, dtype=int)
    sampled_indices[0] = np.random.randint(N)
    min_distances = np.full(N, np.inf)

    for i in range(1, num_samples):
        last_sampled_point = points[sampled_indices[i - 1]]
        distances = np.sum((points - last_sampled_point) ** 2, axis=1)
        min_distances = np.minimum(min_distances, distances)
        sampled_indices[i] = np.argmax(min_distances)
    return points[sampled_indices]


class Sequence(object):
    def __init__(self, data_root, id_list, intrinsic_matrix, num_points=4096, cano_idx=0):
        # data_root: path to the data folder
        # id_list: list of frame indices
        self.num_points = num_points
        self.cano_idx = cano_idx
        self.pc_list = []
        depth_dir = os.path.join(data_root, 'depth')
        mask_dir = os.path.join(data_root, 'mask')
        for id in id_list:
            depth_path =  os.path.join(depth_dir, f'{id:05d}.npy')
            mask_path = os.path.join(mask_dir, f'{id:05d}.png')
            depth_map = np.load(depth_path)
            mask = imageio.imread(mask_path)  
            pc = depth_to_pc(depth_map, intrinsic_matrix, mask)
            self.pc_list.append(pc)
        cano_pc = self.pc_list[self.cano_idx]
        centroid, scale = normalize_pc(cano_pc)
        self.centroid = centroid
        self.scale = scale

    def __len__(self):
        return 1

    def __getitem__(self, item):
        complete_pc_list = []
        for pc in self.pc_list:
            # farthest point sampling
            # pc = farthest_point_sampling(pc, self.num_points)
            indices = np.random.choice(pc.shape[0], self.num_points, replace=False)
            pc = pc[indices]
            complete_pc_list.append(pc)
        complete_pc_list = np.stack(complete_pc_list).astype('float32')
        cano_pc = complete_pc_list[self.cano_idx]
        pc_list = np.concatenate((complete_pc_list[:self.cano_idx, :], complete_pc_list[self.cano_idx+1:, :]), axis=0)
        sample = {'cano_pc': cano_pc,
                  'pc_list': pc_list,
                  'complete_pc_list': complete_pc_list}
        return sample