#!/usr/bin/env python3
"""
Create promotional GIF showing side-by-side comparison of:
- Partial input point cloud (original colors)
- Ground truth point cloud (original colors)  
- Predicted point cloud (distance-colored with jet colormap)

Generates a horizontal 1x3 layout with 360° synchronized rotation.
"""

import numpy as np
import open3d as o3d
import argparse
from pathlib import Path
import imageio
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


def render_pointcloud_view(pcd, width, height, camera_params, bg_color=(1, 1, 1), point_size=5, z_offset_ratio=0.0):
    """
    Render a point cloud from a specific camera viewpoint using matplotlib.
    
    Args:
        pcd: Open3D point cloud
        width: Image width
        height: Image height
        camera_params: Dictionary with 'lookat', 'front', 'up', 'distance', 'angle'
        bg_color: Background color (RGB, 0-1 range)
        point_size: Size of points in the scatter plot
        z_offset_ratio: Ratio to shift Z-axis limits (positive = shift view up, object appears lower)
    
    Returns:
        numpy array of rendered image (RGB, 0-255)
    """
    # Get point cloud data
    points = np.asarray(pcd.points)
    
    # Get colors if available
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.ones_like(points) * 0.5  # Gray default
    
    # Create figure with correct aspect ratio
    dpi = 100
    figsize = (width / dpi, height / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=bg_color)
    ax = fig.add_subplot(111, projection='3d')
    
    # Set background color
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)
    
    # Plot points with configurable size
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=colors, s=point_size, marker='.', alpha=1.0, linewidths=0)
    
    # Set view angle
    angle = camera_params.get('angle', 0)
    ax.view_init(elev=0, azim=angle)
    
    # Set axis limits based on point cloud bounds (tighter zoom)
    center = camera_params['lookat']
    distance = camera_params.get('distance', 0.5)
    limit = distance * 0.3  # Tighter zoom (was 0.5)
    
    # Apply Z-offset to shift view (positive offset = view shifts up, object appears lower)
    z_shift = limit * z_offset_ratio
    
    ax.set_xlim([center[0] - limit, center[0] + limit])
    ax.set_ylim([center[1] - limit, center[1] + limit])
    ax.set_zlim([center[2] - limit + z_shift, center[2] + limit + z_shift])
    
    # Remove axes for cleaner look
    ax.set_axis_off()
    
    # Make axes equal
    ax.set_box_aspect([1, 1, 1])
    
    # Adjust layout to remove margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Render to image
    fig.canvas.draw()
    # Use buffer_rgba instead of deprecated tostring_rgb
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    # Convert RGBA to RGB
    img_array = img_array[:, :, :3]
    
    plt.close(fig)
    
    return img_array


def compute_alignment_rotation(vec_from, vec_to):
    """
    Compute rotation matrix that aligns vec_from with vec_to.
    
    Args:
        vec_from: Source vector (3D, will be normalized)
        vec_to: Target vector (3D, will be normalized)
    
    Returns:
        3x3 rotation matrix
    """
    # Normalize vectors
    v1 = vec_from / np.linalg.norm(vec_from)
    v2 = vec_to / np.linalg.norm(vec_to)
    
    # If vectors are already aligned, return identity
    if np.allclose(v1, v2):
        return np.eye(3)
    
    # If vectors are opposite, return 180° rotation around perpendicular axis
    if np.allclose(v1, -v2):
        # Find a perpendicular vector
        if abs(v1[0]) < 0.9:
            perp = np.cross(v1, [1, 0, 0])
        else:
            perp = np.cross(v1, [0, 1, 0])
        perp = perp / np.linalg.norm(perp)
        # Rodrigues' rotation formula for 180° around perp
        return 2 * np.outer(perp, perp) - np.eye(3)
    
    # General case: use Rodrigues' rotation formula
    # Rotation axis
    axis = np.cross(v1, v2)
    axis = axis / np.linalg.norm(axis)
    
    # Rotation angle
    cos_angle = np.dot(v1, v2)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    # Rodrigues' formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    return R


def find_longest_axis(points, method='pca_upright', mesh_path=None):
    """
    Find the longest axis of a point cloud or mesh using various methods.
    
    Args:
        points: Numpy array (N, 3) - point cloud coordinates
        method: Method to use:
                - 'mesh_obb': Oriented bounding box from closed mesh (BEST - uses complete geometry)
                - 'pca_upright': PCA to find natural axis, ensure it points upward
                - 'pca': Principal Component Analysis (largest variance, may point any direction)
                - 'bounding_box': Axis-aligned bounding box longest edge (uses tilted coords)
                - 'z_axis': Simply use vertical axis (assumes upright fruit)
                - 'hybrid': Bounding box with Z-bias for near-ties
        mesh_path: Path to mesh file (required for 'mesh_obb' method)
    
    Returns:
        Longest axis direction (normalized vector)
    """
    if method == 'mesh_obb':
        # Use oriented bounding box from closed mesh (most accurate)
        if mesh_path is None:
            raise ValueError("mesh_path is required for 'mesh_obb' method")
        
        try:
            # Load the mesh
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            
            if not mesh.has_vertices():
                print(f"Warning: Could not load mesh from {mesh_path}, falling back to PCA")
                return find_longest_axis(points, method='pca_upright', mesh_path=None)
            
            # Compute oriented bounding box
            obb = mesh.get_oriented_bounding_box()
            
            # Get the OBB's rotation matrix and extent
            # The extent gives the size in each principal direction
            extent = np.array(obb.extent)  # [width, height, depth] in OBB coordinates
            rotation = np.array(obb.R)  # 3x3 rotation matrix
            
            # Find which principal axis is longest
            longest_idx = np.argmax(extent)
            
            # Get the corresponding axis in world coordinates
            # The rotation matrix columns are the OBB's principal axes
            longest_axis = rotation[:, longest_idx]
            
            # Ensure it points upward (positive Z component)
            if longest_axis[2] < 0:
                longest_axis = -longest_axis
            
            print(f"  Mesh OBB extents: {extent}")
            print(f"  Using axis {longest_idx} (length: {extent[longest_idx]:.3f}m)")
            
            return longest_axis
            
        except Exception as e:
            print(f"Warning: Failed to load mesh ({e}), falling back to PCA")
            return find_longest_axis(points, method='pca_upright', mesh_path=None)
    
    elif method == 'z_axis':
        # Simplest: assume fruits are vertical
        return np.array([0, 0, 1])
    
    elif method == 'pca' or method == 'pca_upright':
        # Find the actual longest physical axis using PCA
        centered = points - points.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        longest_axis_idx = np.argmax(eigenvalues)
        longest_axis = eigenvectors[:, longest_axis_idx].real
        longest_axis = longest_axis / np.linalg.norm(longest_axis)
        
        if method == 'pca_upright':
            # Ensure the axis points upward (positive Z component)
            # This makes tilted fruits rotate around their natural vertical axis
            if longest_axis[2] < 0:
                longest_axis = -longest_axis
        
        return longest_axis
    
    elif method == 'bounding_box':
        # Use axis-aligned bounding box (WARNING: uses tilted coordinates)
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        extents = max_coords - min_coords
        
        # Find dimension with largest extent
        longest_dim = np.argmax(extents)
        longest_axis = np.zeros(3)
        longest_axis[longest_dim] = 1.0
        
        return longest_axis
    
    elif method == 'hybrid':
        # Bounding box with Z-axis preference for near-ties
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        extents = max_coords - min_coords
        
        # If Z-extent is within 80% of the max extent, prefer Z
        max_extent = extents.max()
        z_extent = extents[2]
        
        if z_extent >= 0.8 * max_extent:
            # Z is close enough to max, use it (fruits are typically vertical)
            return np.array([0, 0, 1])
        else:
            # Otherwise use the actual longest dimension
            longest_dim = np.argmax(extents)
            longest_axis = np.zeros(3)
            longest_axis[longest_dim] = 1.0
            return longest_axis
    
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_camera_orbit_params(center, radius, angle_degrees, rotation_axis=(0, 0, 1), zoom=0.7):
    """
    Calculate camera parameters for orbiting around a point.
    
    Args:
        center: Point to look at (3D)
        radius: Distance from center
        angle_degrees: Rotation angle around the rotation axis
        rotation_axis: Axis to rotate around (default Z-axis)
        zoom: Zoom level (controls camera distance)
    
    Returns:
        Dictionary with camera parameters
    """
    angle_rad = np.radians(angle_degrees)
    
    # Adjust radius based on zoom (lower zoom = farther away)
    adjusted_radius = radius / max(zoom, 0.1)
    
    # Normalize rotation axis
    rotation_axis = np.array(rotation_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # Find a perpendicular vector to the rotation axis for the orbit plane
    if abs(rotation_axis[2]) < 0.9:
        perpendicular = np.cross(rotation_axis, [0, 0, 1])
    else:
        perpendicular = np.cross(rotation_axis, [1, 0, 0])
    perpendicular = perpendicular / np.linalg.norm(perpendicular)
    
    # Create another perpendicular vector
    perpendicular2 = np.cross(rotation_axis, perpendicular)
    perpendicular2 = perpendicular2 / np.linalg.norm(perpendicular2)
    
    # Camera position on orbit (rotate in the plane perpendicular to rotation_axis)
    camera_offset = (perpendicular * np.cos(angle_rad) + perpendicular2 * np.sin(angle_rad)) * adjusted_radius
    camera_pos = center + camera_offset
    
    # Front vector points from camera to center
    front = center - camera_pos
    front = front / np.linalg.norm(front)
    
    return {
        'lookat': center,
        'front': front,
        'up': rotation_axis,  # Up vector is the rotation axis
        'distance': adjusted_radius,
        'zoom': zoom,
        'angle': angle_degrees  # For matplotlib azimuth
    }


def add_text_to_image(img_array, text, position='top', font_size=15, color=(0, 0, 0)):
    """
    Add text overlay to image using matplotlib (works without system fonts).
    
    Args:
        img_array: Numpy array (H, W, 3)
        text: Text to add
        position: 'top' or 'bottom'
        font_size: Font size in points (matplotlib units)
        color: Text color (RGB, 0-255)
    
    Returns:
        Modified image array
    """
    # Convert color to 0-1 range for matplotlib
    color_normalized = tuple(c / 255.0 for c in color)
    
    # Create figure matching the image size
    height, width = img_array.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Display the image as background
    ax.imshow(img_array)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Flip Y axis to match image coordinates
    ax.axis('off')
    
    # Calculate text position
    x = width / 2
    if position == 'top':
        y = font_size  # Distance from top
        va = 'top'
    else:
        y = height - font_size  # Distance from bottom
        va = 'bottom'
    
    # Add text with outline for visibility
    outline_color = 'white' if sum(color) < 384 else 'black'
    
    # Draw outline
    for dx in [-2, 0, 2]:
        for dy in [-2, 0, 2]:
            if dx != 0 or dy != 0:
                ax.text(x + dx, y + dy, text,
                       fontsize=font_size, fontweight='bold',
                       ha='center', va=va,
                       color=outline_color)
    
    # Draw main text
    ax.text(x, y, text,
           fontsize=font_size, fontweight='bold',
           ha='center', va=va,
           color=color_normalized)
    
    # Render to array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_array = img_array[:, :, :3]  # Drop alpha
    
    plt.close(fig)
    
    return img_array


def create_colorbar_image(colorbar_width, total_height, max_distance_m=0.02, cmap_name='YlOrRd',
                         bg_color=(1, 1, 1), font_size=10):
    """
    Create a VERTICAL colorbar image as a separate panel with its own background using matplotlib.
    
    Args:
        colorbar_width: Width of the colorbar panel
        total_height: Image height
        max_distance_m: Maximum distance in meters for the color scale
        cmap_name: Colormap name (should match prediction coloring)
        bg_color: Background color (RGB, 0-1)
        font_size: Font size for labels
    
    Returns:
        Colorbar image as numpy array (H, W, 3)
    """
    # Create figure with exact dimensions
    dpi = 100
    fig = plt.figure(figsize=(colorbar_width/dpi, total_height/dpi), dpi=dpi, facecolor=bg_color)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, colorbar_width)
    ax.set_ylim(0, total_height)
    ax.axis('off')
    
    # Vertical colorbar dimensions - thinner bar
    bar_width = 30
    bar_height = int(total_height * 0.6)
    margin_left = 20
    margin_top = int(total_height * 0.3)  # Reduced from 0.2 to move colorbar lower
    
    # Position in the colorbar panel
    x_start = margin_left
    y_start = total_height - margin_top  # Flip Y for matplotlib coordinates (origin at bottom)
    y_end = y_start - bar_height
    
    # Create PERFECTLY SMOOTH gradient using matplotlib directly
    # This ensures pixel-perfect rendering with no artifacts
    cmap = plt.get_cmap(cmap_name)
    
    # Create high-resolution gradient for ultra-smooth appearance
    # Using more samples than pixels ensures smooth interpolation
    gradient = np.linspace(1, 0, bar_height * 4).reshape(-1, 1)  # 4x resolution
    gradient_2d = np.tile(gradient, (1, bar_width))  # Extend horizontally
    
    # Use matplotlib's imshow with 'bilinear' interpolation for smooth gradient
    # The colorbar is static, so interpolation won't cause frame-to-frame variations
    extent = [x_start, x_start + bar_width, y_end, y_start]
    ax.imshow(gradient_2d, aspect='auto', cmap=cmap, extent=extent, interpolation='bilinear')
    
    # Add labels with matplotlib (consistent with titles)
    label_x = x_start + bar_width + 12
    
    # Max label at top (high distance = red)
    max_label = f"{max_distance_m*1000:.0f} mm"
    ax.text(label_x, y_start, max_label, fontsize=font_size, fontweight='bold',
            ha='left', va='top', color='black')
    
    # Middle label
    mid_label = f"{max_distance_m*1000/2:.0f} mm"
    ax.text(label_x, (y_start + y_end) / 2, mid_label, fontsize=font_size, fontweight='bold',
            ha='left', va='center', color='black')
    
    # Min label at bottom (low distance = green/blue)
    min_label = "0 mm"
    ax.text(label_x, y_end, min_label, fontsize=font_size, fontweight='bold',
            ha='left', va='bottom', color='black')
    
    # Title above colorbar - single multi-line text
    title_x = x_start + bar_width / 2 + 25  # Shift title to the right to avoid overlap
    title_y = y_start + font_size * 3.0  # Above the colorbar
    ax.text(title_x, title_y, "Chamfer\ndistance", fontsize=font_size + 4, fontweight='bold',
            ha='center', va='bottom', color='black', linespacing=1.5)
    
    # Render to array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_array = img_array[:, :, :3]  # Drop alpha
    
    plt.close(fig)
    
    return img_array


def add_colorbar_panel(combined_img_array, colorbar_img_array):
    """
    Append colorbar panel to the right of the combined image.
    
    Args:
        combined_img_array: Numpy array (H, W, 3) - combined 3-panel image
        colorbar_img_array: Colorbar panel image (H, W, 3)
    
    Returns:
        Modified image array with colorbar panel on the right
    """
    # Simply concatenate horizontally
    return np.hstack([combined_img_array, colorbar_img_array])


def quantize_frames_with_colorbar_priority(frames, colors=256):
    """
    Quantize frames with a global palette that prioritizes colorbar gradient colors.
    This prevents banding and shifting in the colorbar across frames.
    """
    if not frames:
        return frames
    
    # Sample colors from multiple frames to build representative palette
    sample_indices = np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int)
    
    # Collect pixel samples from frames
    all_pixels = []
    for idx in sample_indices:
        frame = frames[idx]
        # Sample more heavily from the right side (where colorbar is)
        h, w = frame.shape[:2]
        colorbar_region = frame[:, int(w * 0.8):, :]  # Right 20% (colorbar area)
        main_region = frame[:, :int(w * 0.8):10, :]   # Subsample main area
        
        all_pixels.append(colorbar_region.reshape(-1, 3))
        all_pixels.append(main_region.reshape(-1, 3))
    
    # Create palette from sampled pixels
    palette_data = np.vstack(all_pixels)
    # Limit to reasonable size for quantization
    if len(palette_data) > 100000:
        indices = np.random.choice(len(palette_data), 100000, replace=False)
        palette_data = palette_data[indices]
    
    # Create image from palette data and quantize
    h = int(np.sqrt(len(palette_data)))
    w = len(palette_data) // h
    palette_data = palette_data[:h*w].reshape(h, w, 3)
    palette_img = Image.fromarray(palette_data.astype(np.uint8))
    palette_img = palette_img.quantize(colors=colors, method=Image.MEDIANCUT)
    
    # Quantize all frames with this palette
    quantized_frames = []
    for frame in frames:
        img = Image.fromarray(frame.astype(np.uint8))
        # Use the shared palette with Floyd-Steinberg dithering for smooth gradients
        quantized = img.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
        # Convert back to RGB for imageio
        quantized_rgb = quantized.convert('RGB')
        quantized_frames.append(np.array(quantized_rgb))
    
    return quantized_frames


def combine_images_horizontal(img1, img2, img3):
    """
    Combine three images horizontally.
    
    Args:
        img1, img2, img3: Numpy arrays (H, W, 3)
    
    Returns:
        Combined image array
    """
    return np.hstack([img1, img2, img3])


def create_promotional_gif(partial_path, gt_path, pred_path, output_path,
                           num_frames=60, fps=30, 
                           resolution_per_panel=(640, 640),
                           zoom=0.8, bg_color=(1, 1, 1),
                           titles=None, show_colorbar=True, max_distance=0.02,
                           point_size=5, title_font_size=10, rotation_method='pca_upright',
                           pred_mesh_path=None):
    """
    Create promotional GIF with side-by-side point cloud comparison.
    
    Args:
        partial_path: Path to partial input point cloud (optional)
        gt_path: Path to ground truth point cloud (optional)
        pred_path: Path to predicted point cloud (optional, should already be colored)
        output_path: Path to save output GIF
        num_frames: Number of frames for 360° rotation
        fps: Frames per second
        resolution_per_panel: (width, height) for each panel
        zoom: Camera zoom level
        bg_color: Background color (RGB, 0-1)
        titles: List of titles (auto-adjusted based on provided point clouds)
        show_colorbar: Whether to add colorbar to prediction panel
        max_distance: Maximum distance for colorbar scale (meters)
        point_size: Size of points in scatter plot (default: 5)
        title_font_size: Font size for panel titles (default: 150)
        rotation_method: Method to find rotation axis ('mesh_obb', 'pca_upright', 'pca', 'bounding_box', 'z_axis', 'hybrid')
        pred_mesh_path: Path to predicted mesh file (required for 'mesh_obb' method)
    """
    # Load point clouds (only those provided)
    print("Loading point clouds...")
    point_clouds = []
    default_titles = []
    
    if partial_path:
        partial_pcd = o3d.io.read_point_cloud(str(partial_path))
        point_clouds.append(('partial', partial_pcd))
        default_titles.append('Partial Input')
        print(f"  Partial: {len(partial_pcd.points)} points")
    
    if gt_path:
        gt_pcd = o3d.io.read_point_cloud(str(gt_path))
        point_clouds.append(('gt', gt_pcd))
        default_titles.append('Ground Truth')
        print(f"  GT:      {len(gt_pcd.points)} points")
    
    if pred_path:
        pred_pcd = o3d.io.read_point_cloud(str(pred_path))
        point_clouds.append(('pred', pred_pcd))
        default_titles.append('Prediction')
        print(f"  Pred:    {len(pred_pcd.points)} points")
    
    if not point_clouds:
        raise ValueError("At least one point cloud must be provided")
    
    # Use provided titles or defaults
    if titles is None:
        titles = default_titles
    
    # Only show colorbar if prediction is included
    show_colorbar = show_colorbar and pred_path is not None
    
    # Calculate bounding box center and size for camera setup
    # Use GT if available (most complete), otherwise use first available point cloud
    reference_pcd = gt_pcd if gt_path else point_clouds[0][1]
    reference_points = np.asarray(reference_pcd.points)
    center = reference_points.mean(axis=0)
    bbox_size = reference_points.max(axis=0) - reference_points.min(axis=0)
    radius = np.linalg.norm(bbox_size) * 1.5  # Camera distance
    
    # Find the longest axis of the fruit (for natural rotation)
    # Use predicted mesh if available and requested
    longest_axis = find_longest_axis(reference_points, method=rotation_method, mesh_path=pred_mesh_path)
    
    print(f"\nCamera setup:")
    print(f"  Center: {center}")
    print(f"  Radius: {radius:.3f}")
    print(f"  Zoom: {zoom}")
    print(f"  Rotation method: {rotation_method}")
    if pred_mesh_path and rotation_method == 'mesh_obb':
        print(f"  Using mesh: {pred_mesh_path}")
    print(f"  Rotation axis: {longest_axis}")
    
    # Transform all point clouds to align rotation axis with Z
    # This way matplotlib's azimuth rotation will work correctly
    rotation_matrix = compute_alignment_rotation(longest_axis, np.array([0, 0, 1]))
    print(f"  Aligning rotation axis with Z-axis...")
    
    # Rotate point clouds (Open3D rotate is around origin, center parameter is for visualization only)
    # Only rotate the point clouds that were loaded
    if partial_path:
        partial_pcd = partial_pcd.translate(-center).rotate(rotation_matrix).translate(center)
    if gt_path:
        gt_pcd = gt_pcd.translate(-center).rotate(rotation_matrix).translate(center)
    if pred_path:
        pred_pcd = pred_pcd.translate(-center).rotate(rotation_matrix).translate(center)
    
    width, height = resolution_per_panel
    
    # Calculate Z-offset ratio to shift view for title space
    # Title takes title_font_size pixels at top, so we shift the view up (object appears lower)
    # Use a much larger multiplier to make the effect visible
    title_height_ratio = title_font_size / height  # Title as fraction of image height
    z_offset_ratio = title_height_ratio * 10.0  # Large multiplier for visible shift (was 1.5, too small!)
    
    print(f"  Z-offset ratio for title space: {z_offset_ratio:.3f} (title: {title_font_size}px, height: {height}px)")
    
    frames = []
    
    # Create colorbar panel once (will be reused for all frames)
    colorbar_width = 150  # Width of the separate colorbar panel
    colorbar_img = None
    if show_colorbar:
        print("\nCreating fixed vertical colorbar panel...")
        colorbar_img = create_colorbar_image(colorbar_width, height, max_distance_m=max_distance, 
                                            cmap_name='YlOrRd', bg_color=bg_color)
    
    print(f"\nRendering {num_frames} frames...")
    for i in range(num_frames):
        angle = (360 / num_frames) * i
        
        if i % 10 == 0:
            print(f"  Frame {i+1}/{num_frames} (angle: {angle:.1f}°)")
        
        # Calculate camera parameters for this angle (rotate around Z-axis since we aligned the point clouds)
        camera_params = calculate_camera_orbit_params(center, radius, angle, rotation_axis=np.array([0, 0, 1]), zoom=zoom)
        
        # Render each point cloud that was provided
        rendered_images = []
        for idx, (pcd_type, pcd) in enumerate(point_clouds):
            img = render_pointcloud_view(pcd, width, height, camera_params, bg_color, point_size, z_offset_ratio)
            img = add_text_to_image(img, titles[idx], position='top', font_size=title_font_size)
            rendered_images.append(img)
        
        # Combine panels horizontally
        if len(rendered_images) == 1:
            combined = rendered_images[0]
        elif len(rendered_images) == 2:
            combined = np.hstack(rendered_images)
        else:  # 3 or more
            combined = np.hstack(rendered_images)
        
        # Add colorbar panel on the right (as a 4th panel with its own background)
        if show_colorbar and colorbar_img is not None:
            combined = add_colorbar_panel(combined, colorbar_img)
        
        frames.append(combined)
    
    print(f"\nSaving GIF to {output_path}...")
    print(f"  Total frames: {len(frames)}")
    print(f"  Resolution: {frames[0].shape[1]}x{frames[0].shape[0]}")
    print(f"  FPS: {fps}")
    
    # Quantize frames with colorbar-aware palette to prevent banding and shifting
    print("  Quantizing frames with stable palette...")
    quantized_frames = quantize_frames_with_colorbar_priority(frames, colors=256)
    
    imageio.mimsave(output_path, quantized_frames, fps=fps, loop=0)
    
    # Print file size
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\n✓ GIF created successfully!")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Duration: {num_frames/fps:.1f} seconds")


def main():
    parser = argparse.ArgumentParser(
        description='Create promotional GIF from point clouds (single or multiple panels)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single point cloud (GT only) with mesh-based rotation
  python create_promo_gif.py \\
    --gt data/kmelon_centered/S1/laser/fruit.ply \\
    --mesh data/kmelon_centered/S1/laser/fruit.ply \\
    --rotation-method mesh_obb \\
    --output S1_gt.gif
  
  # Two point clouds (partial + GT)
  python create_promo_gif.py --partial data/S1_partial.ply --gt data/S1_gt.ply --output S1_comparison.gif
  
  # All three (partial + GT + prediction with colorbar)
  python create_promo_gif.py --partial data/S1_partial.ply --gt data/S1_gt.ply --pred data/S1_pred.ply --output S1_full.gif
        """
    )
    
    # Input point clouds (at least one required)
    parser.add_argument('--partial', type=str, default=None,
                       help='Path to partial input point cloud (.ply)')
    parser.add_argument('--gt', type=str, default=None,
                       help='Path to ground truth point cloud (.ply)')
    parser.add_argument('--pred', type=str, default=None,
                       help='Path to predicted point cloud (.ply, should be distance-colored)')
    
    # Output
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output GIF path')
    
    # Animation settings
    parser.add_argument('--frames', type=int, default=60,
                       help='Number of frames for 360° rotation (default: 60)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    
    # Visual settings
    parser.add_argument('--width', type=int, default=640,
                       help='Width of each panel (default: 640, try 400 for larger text)')
    parser.add_argument('--height', type=int, default=640,
                       help='Height of each panel (default: 640, try 400 for larger text)')
    parser.add_argument('--zoom', type=float, default=1.5,
                       help='Camera zoom level (default: 1.5, higher=closer)')
    parser.add_argument('--point-size', type=float, default=5.0,
                       help='Size of points in scatter plot (default: 5.0)')
    parser.add_argument('--title-font-size', type=int, default=15,
                       help='Font size for panel titles (default: 15)')
    
    # Colorbar settings (only shown if --pred is provided)
    parser.add_argument('--no-colorbar', action='store_true',
                       help='Do not add colorbar to prediction panel (colorbar auto-disabled if --pred not provided)')
    parser.add_argument('--max-distance', type=float, default=0.02,
                       help='Maximum distance for colorbar scale in meters (default: 0.02)')
    
    # Rotation settings
    parser.add_argument('--rotation-method', type=str, default='pca_upright',
                       choices=['mesh_obb', 'pca_upright', 'pca', 'bounding_box', 'z_axis', 'hybrid'],
                       help='Method to find rotation axis (default: pca_upright). '
                            'mesh_obb: Use mesh oriented bbox (BEST for closed meshes, requires --mesh); '
                            'pca_upright: PCA natural axis, upward-facing; '
                            'pca: PCA any direction; bounding_box: longest dimension; '
                            'z_axis: vertical; hybrid: Z-biased bounding box')
    parser.add_argument('--mesh', type=str, default=None,
                       help='Path to mesh file for rotation axis calculation (use with --rotation-method mesh_obb). '
                            'Can be GT mesh, predicted mesh, or any closed mesh of the fruit.')
    parser.add_argument('--pred-mesh', type=str, default=None,
                       help='[DEPRECATED: use --mesh instead] Path to predicted mesh file')
    
    # Custom titles
    parser.add_argument('--titles', nargs='+', default=None,
                       help='Custom titles for panels (number must match provided point clouds)')
    
    args = parser.parse_args()
    
    # Validate that at least one point cloud is provided
    if not any([args.partial, args.gt, args.pred]):
        print("Error: At least one point cloud must be provided (--partial, --gt, or --pred)")
        parser.print_help()
        return 1
    
    # Backward compatibility: use --pred-mesh if --mesh not provided
    mesh_path = args.mesh if args.mesh else args.pred_mesh
    if args.pred_mesh and not args.mesh:
        print("Warning: --pred-mesh is deprecated, use --mesh instead")
    
    # Validate mesh_obb requires mesh
    if args.rotation_method == 'mesh_obb' and not mesh_path:
        print("Error: --rotation-method mesh_obb requires --mesh argument")
        return 1
    
    # Validate input files exist
    for path_arg, path_val in [('--partial', args.partial), ('--gt', args.gt), ('--pred', args.pred), ('--mesh', mesh_path)]:
        if path_val and not Path(path_val).exists():
            print(f"Error: File not found: {path_val} (specified by {path_arg})")
            return 1
    
    try:
        create_promotional_gif(
            partial_path=args.partial,
            gt_path=args.gt,
            pred_path=args.pred,
            output_path=args.output,
            num_frames=args.frames,
            fps=args.fps,
            resolution_per_panel=(args.width, args.height),
            zoom=args.zoom,
            titles=args.titles,
            show_colorbar=not args.no_colorbar,
            max_distance=args.max_distance,
            point_size=args.point_size,
            title_font_size=args.title_font_size,
            rotation_method=args.rotation_method,
            pred_mesh_path=mesh_path
        )
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

