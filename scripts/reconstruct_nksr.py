import numpy as np
import cv2
import open3d as o3d
import os
from glob import glob





import torch
import numpy as np
from pycg import vis, exp
from nksr import Reconstructor, utils, fields, get_estimate_normal_preprocess_fn
import gzip
import glob
from natsort import natsorted
import open3d as o3d



UPSAMPLE = 8

def upsample_data(depth, mask, rgb):
    """Upsample all input data to double resolution using nearest neighbor interpolation."""
    # Get target dimensions (double the original)
    target_height = depth.shape[0] * UPSAMPLE
    target_width = depth.shape[1] * UPSAMPLE
    
    # Upsample depth (using cv2 resize with nearest neighbor)
    depth_upsampled = cv2.resize(depth, (target_width, target_height), 
                                interpolation=cv2.INTER_NEAREST)
    
    # Upsample mask
    mask_upsampled = cv2.resize(mask, (target_width, target_height), 
                               interpolation=cv2.INTER_NEAREST)
    
    # Upsample RGB
    rgb_upsampled = cv2.resize(rgb, (target_width, target_height), 
                              interpolation=cv2.INTER_NEAREST)
    
    return depth_upsampled, mask_upsampled, rgb_upsampled

def load_frame(base_path, frame_num):
    """Load depth, mask, and RGB data for a single frame."""
    # Format frame number to 5 digits
    frame = f"{frame_num:05d}"
    
    depth = np.load(os.path.join(base_path, 'depth', f'{frame}.npy'))
    mask = cv2.imread(os.path.join(base_path, 'mask', f'{frame}.png'), cv2.IMREAD_GRAYSCALE)
    rgb = cv2.imread(os.path.join(base_path, 'rgb', f'{frame}.jpg'))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # Upsample all data
    depth, mask, rgb = upsample_data(depth, mask, rgb)
    
    # Increase erosion by using a larger kernel and more iterations
    kernel = np.ones((32,32), np.uint8)  # Increased kernel size from 3x3 to 5x5
    mask = cv2.erode(mask, kernel, iterations=4)  # Increased iterations from 2 to 4
    
    return depth, mask, rgb

def create_camera_intrinsic(rgb_shape):
    """Create camera intrinsic matrix using the provided parameters."""
    # Double the intrinsic parameters to match the upsampled resolution
    fx = (1373.8807373046875 / 7.5) * UPSAMPLE
    fy = (1373.8807373046875 / 7.5) * UPSAMPLE
    cx = (719.4837646484375 / 7.5) * UPSAMPLE
    cy = (963.06695556640625 / 7.5) * UPSAMPLE
    
    return o3d.camera.PinholeCameraIntrinsic(
        width=int(rgb_shape[1]),
        height=int(rgb_shape[0]),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )

def create_point_cloud(depth, rgb, intrinsic, mask=None):
    """Create a colored point cloud from depth and RGB images."""
    # Convert depth and rgb to Open3D format
    depth_o3d = o3d.geometry.Image(depth)
    rgb_o3d = o3d.geometry.Image(rgb)
    
    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, 
        depth_o3d,
        depth_scale=2.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        intrinsic
    )
    
    # Apply mask if provided
    if mask is not None:
        # Convert mask to boolean array
        mask_bool = mask > 0
        # Get points as numpy array
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        # Apply mask
        pcd.points = o3d.utility.Vector3dVector(points[mask_bool.flatten()])
        pcd.colors = o3d.utility.Vector3dVector(colors[mask_bool.flatten()])
    
    return pcd

def process_all_frames(base_path, start_frame=0, end_frame=18):
    """Process multiple frames and combine them into a single point cloud."""
    combined_pcd = None
    
    for frame_num in range(start_frame, end_frame + 1):
        print(f"Processing frame {frame_num:05d}")
        
        # Load frame data
        depth, mask, rgb = load_frame(base_path, frame_num)
        
        # Create camera intrinsic (using first frame's dimensions)
        if frame_num == start_frame:
            intrinsic = create_camera_intrinsic(rgb.shape)
        
        # Create point cloud for this frame
        pcd = create_point_cloud(depth, rgb, intrinsic, mask)
        
        # Downsample and remove outliers for each frame
        #pcd = pcd.voxel_down_sample(voxel_size=0.01)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Combine with existing point cloud
        if combined_pcd is None:
            combined_pcd = pcd
        else:
            combined_pcd += pcd
    
    # Final downsampling of combined point cloud to remove redundant points
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.001)
    
    return combined_pcd

def visualize_point_cloud(pcd):
    """Visualize the point cloud."""
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[0, 0, 0]
    )
    
    # Visualize
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # Set the base path
    base_path = "data/attempt1/laptop2"
    
    # Process all frames and get combined point cloud
    combined_pcd = process_all_frames(base_path, start_frame=0, end_frame=1)
    
    # Visualize
    visualize_point_cloud(combined_pcd)

    


    #
    # Run NKSR
    #


    device = torch.device("cuda:0")
    #('cpu')#

    #combined_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))


    input_xyz = torch.from_numpy(np.asarray(combined_pcd.points)).float().to(device)
    input_normal = torch.from_numpy(np.asarray(combined_pcd.normals)).float().to(device)
    input_color = torch.from_numpy(np.asarray(combined_pcd.colors)).float().to(device)
    sensors = torch.zeros((len(combined_pcd.points), 3), dtype=float).to(device)

    nksr = Reconstructor(device)

    nksr.chunk_tmp_device = torch.device("cpu")

    field = nksr.reconstruct(input_xyz, 
                             #normal=input_normal,
                             sensor=sensors, 
                             detail_level=1.0, 
                             preprocess_fn=get_estimate_normal_preprocess_fn(64, 85.0),
                             fused_mode=True,
                             approx_kernel_grad=True,
                             solver_tol=1e-4)
    
    field.set_texture_field(fields.PCNNField(input_xyz, input_color))

    mesh = field.extract_dual_mesh(mise_iter=1, max_points=2 ** 18) #max_points=2 ** 22
    mesh = vis.mesh(mesh.v, mesh.f, color=mesh.c)
    vis.to_file(mesh, "mesh.ply")

    print('test')
    #vis.show_3d([mesh], [combined_pcd])