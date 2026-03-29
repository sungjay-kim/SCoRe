# Step 5: Visualization

Generate visualizations for presentations, papers, and debugging.

---

## Overview

Create various visualizations:
1. **SDF Visualizations** - Cross-sections and 3D rendering
2. **Promotional GIFs** - Rotating meshes for presentations
3. **Error Visualizations** - Color-coded error maps
4. **Comparison Plots** - Side-by-side comparisons

---

## Quick Start

### Generate Rotating GIF

```bash
python3 create_promo_gif.py \
    --mesh_file experiments/.../inf/meshes/S1_0_mesh.ply \
    --output output.gif \
    --duration 3.0 \
    --fps 30
```

### Visualize SDF Cross-Section

```bash
python3 visualize_sdf_simple.py \
    --experiment deepsdf/experiments/kmelon_latent32 \
    --checkpoint 301 \
    --sample S1 \
    --output sdf_visualization.png
```

---

## SDF Visualization

### Simple Cross-Section Viewer

Visualize 2D cross-sections of the SDF field:

```bash
python3 visualize_sdf_simple.py \
    --experiment deepsdf/experiments/kmelon_latent32 \
    --checkpoint 301 \
    --sample S1 \
    --slice_axis z \
    --slice_position 0.0 \
    --output sdf_slice.png
```

**Parameters:**
- `--experiment`: DeepSDF experiment directory
- `--checkpoint`: Epoch number
- `--sample`: Sample ID to visualize
- `--slice_axis`: Axis for cross-section (x/y/z)
- `--slice_position`: Position along axis (-1 to 1)
- `--output`: Output image file

**Output:** 2D heatmap showing SDF values
- **Blue**: Inside (negative SDF)
- **Red**: Outside (positive SDF)
- **Green/Yellow**: Near surface (SDF ≈ 0)

### Animated SDF GIF

Create animated GIF sweeping through SDF slices:

```bash
python3 visualize_sdf_gif.py \
    --experiment deepsdf/experiments/kmelon_latent32 \
    --checkpoint 301 \
    --sample S1 \
    --slice_axis z \
    --num_slices 30 \
    --output sdf_animation.gif
```

**Parameters:**
- `--num_slices`: Number of slices in animation
- Other parameters same as simple viewer

**Output:** Animated GIF showing SDF field from different angles

---

## Promotional GIFs

### Create Rotating Mesh GIF

```bash
python3 create_promo_gif.py \
    --mesh_file experiments/.../inf/meshes/S1_0_mesh.ply \
    --output promo.gif \
    --duration 3.0 \
    --fps 30 \
    --elevation 30 \
    --distance 2.0
```

**Parameters:**
- `--mesh_file`: Input mesh file (.ply)
- `--output`: Output GIF filename
- `--duration`: Duration in seconds
- `--fps`: Frames per second
- `--elevation`: Camera elevation angle (degrees)
- `--distance`: Camera distance from object
- `--resolution`: Image resolution (default: 800x600)

**Output:** Smooth rotating GIF of the mesh

### Batch Generate GIFs

Create GIFs for all test results:

```bash
#!/bin/bash
# Generate GIFs for all meshes

MESH_DIR="experiments/kmelon_latent32/20251031_test/inf/meshes"
OUTPUT_DIR="gifs"

mkdir -p ${OUTPUT_DIR}

for mesh in ${MESH_DIR}/*.ply; do
    basename=$(basename ${mesh} .ply)
    echo "Creating GIF for ${basename}..."
    python3 create_promo_gif.py \
        --mesh_file ${mesh} \
        --output ${OUTPUT_DIR}/${basename}.gif \
        --duration 3.0 \
        --fps 30
done
```

### Test GIF Generation

Test script with multiple configurations:

```bash
bash generate_promo_gifs_test.sh
```

This generates GIFs with various camera angles and lighting configurations.

---

## Error Visualization

### Color-Coded Error Maps

Enable during testing to generate colored point clouds:

**Config:**
```json
{
    "save_colored_mesh": true,
    "visualization_colormap": "jet",
    "visualization_max_distance": 0.02
}
```

**Output:** Point clouds colored by distance to GT
- **Blue**: Low error (< 0.5cm)
- **Green/Yellow**: Medium error (0.5-1.5cm)
- **Red**: High error (> 1.5cm)

### Available Colormaps

| Colormap | Description | Best For |
|----------|-------------|----------|
| `jet` | Blue → Green → Yellow → Red | Intuitive error visualization |
| `viridis` | Purple → Green → Yellow | Perceptually uniform, color-blind friendly |
| `plasma` | Purple → Pink → Orange → Yellow | High contrast |
| `turbo` | Enhanced jet | Better perceptual uniformity |
| `RdYlGn_r` | Red → Yellow → Green | Traffic light style |

### Viewing Colored Point Clouds

**CloudCompare:**
```bash
cloudcompare S1_0_colored.ply
```

**MeshLab:**
```bash
meshlab S1_0_colored.ply
```

**Python (Open3D):**
```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("S1_0_colored.ply")
o3d.visualization.draw_geometries([pcd])
```

---

## Comparison Visualizations

### Visualize Colored Point Clouds

Create comparison views with GT and prediction:

```bash
python3 visualize_colored_pointclouds.py \
    --pred_pcd experiments/.../inf/meshes/S1_0_mesh.ply \
    --gt_pcd data/kmelon_centered/S1/laser/fruit.ply \
    --output comparison.png
```

**Output:** Side-by-side visualization with color-coded errors

### Demo Visualization

Create demo visualizations for presentations:

```bash
python3 demo_visualization.py \
    --experiment experiments/kmelon_latent32/20251031_test \
    --samples S1,S2,S3 \
    --output demo_figure.png
```

**Output:** Multi-panel figure showing:
- Input partial point cloud
- Predicted complete mesh
- Ground truth
- Error visualization

---

## Visualization Best Practices

### For Papers

1. **Use perceptually uniform colormaps**
   ```python
   visualization_colormap = "viridis"  # or "plasma"
   ```

2. **Set consistent color scales**
   ```python
   visualization_max_distance = 0.02  # Same for all figures
   ```

3. **High resolution**
   ```python
   resolution = (1920, 1080)  # or higher
   ```

4. **Save as vector graphics when possible**
   ```python
   output_format = "pdf"  # or "svg"
   ```

### For Presentations

1. **Use intuitive colormaps**
   ```python
   visualization_colormap = "jet"  # or "turbo"
   ```

2. **Create animations**
   - Rotating meshes show 3D structure
   - Sweeping SDF slices show internal representation

3. **Side-by-side comparisons**
   - Input vs Output
   - Prediction vs Ground Truth
   - Different models

### For Debugging

1. **Use extreme colors**
   ```python
   visualization_colormap = "RdYlGn_r"  # Red = bad, Green = good
   ```

2. **Visualize intermediate outputs**
   - Partial input
   - Encoded features (if possible)
   - Decoded SDF
   - Final mesh

3. **Check specific problematic regions**
   - Zoom into high-error areas
   - Visualize from multiple angles

---

## Advanced Visualizations

### PCA Axis Visualization

Visualize length/width extraction for size metrics:

```bash
python3 evaluate_size_metrics.py \
    --mesh_dir experiments/.../inf/meshes \
    --gt_csv data/3DKmelonDemo/size_data.csv \
    --output size_evaluation.csv \
    --visualize
```

**Output:** `axis_visualizations/` directory with meshes overlayed with PCA axes
- **Red arrow**: Length axis
- **Green arrow**: Width axis
- **Blue arrow**: Height axis

### Multi-View Renderings

Create multi-view renderings for detailed inspection:

```python
import open3d as o3d

mesh = o3d.io.read_triangle_mesh("S1_0_mesh.ply")
mesh.compute_vertex_normals()

# Create visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(width=1920, height=1080)
vis.add_geometry(mesh)

# Render from different views
views = [
    (0, 0),      # Front
    (90, 0),     # Right
    (180, 0),    # Back
    (270, 0),    # Left
    (0, 90),     # Top
    (0, -90),    # Bottom
]

for azimuth, elevation in views:
    # Set view
    ctr = vis.get_view_control()
    ctr.set_front([np.cos(azimuth), np.sin(azimuth), 0])
    ctr.set_up([0, 0, 1])
    
    # Capture image
    vis.capture_screen_image(f"view_{azimuth}_{elevation}.png")

vis.destroy_window()
```

---

## Batch Visualization Scripts

### Generate All Visualizations

```bash
#!/bin/bash
# Complete visualization pipeline

EXPERIMENT="experiments/kmelon_latent32/20251031_test"
MESH_DIR="${EXPERIMENT}/inf/meshes"
OUTPUT_DIR="visualizations"

mkdir -p ${OUTPUT_DIR}/{gifs,sdf,colored,comparison}

# 1. Create rotating GIFs
echo "Creating GIFs..."
for mesh in ${MESH_DIR}/*_mesh.ply; do
    basename=$(basename ${mesh} _mesh.ply)
    python3 create_promo_gif.py \
        --mesh_file ${mesh} \
        --output ${OUTPUT_DIR}/gifs/${basename}.gif
done

# 2. Create SDF visualizations
echo "Creating SDF visualizations..."
for sample in S1 S2 S3 S4 S5; do
    python3 visualize_sdf_simple.py \
        --experiment deepsdf/experiments/kmelon_latent32 \
        --checkpoint 301 \
        --sample ${sample} \
        --output ${OUTPUT_DIR}/sdf/${sample}_sdf.png
done

# 3. Copy colored point clouds
echo "Copying colored visualizations..."
cp ${EXPERIMENT}/inf/meshes/*_colored.ply ${OUTPUT_DIR}/colored/

# 4. Create comparison plots
echo "Creating comparison plots..."
python3 demo_visualization.py \
    --experiment ${EXPERIMENT} \
    --samples S1,S2,S3 \
    --output ${OUTPUT_DIR}/comparison/demo.png

echo "Visualization complete! Output in ${OUTPUT_DIR}/"
```

---

## Troubleshooting

### Issue: GIF generation fails
**Solutions:**
1. Check if mesh file exists and is valid
2. Install required packages: `pip install imageio pillow`
3. Reduce resolution or fps if too slow
4. Check disk space for output

### Issue: Colored point clouds look wrong
**Solutions:**
1. Verify GT point cloud is properly loaded
2. Check `visualization_max_distance` setting
3. Try different colormap
4. Ensure point clouds are in same coordinate system

### Issue: SDF visualization shows no surface
**Solutions:**
1. Check if latent code was properly loaded
2. Verify decoder checkpoint is correct
3. Check if sample ID exists in training data
4. Try different slice position

### Issue: Renderings are too dark/bright
**Solutions:**
1. Adjust lighting in rendering script
2. Increase/decrease ambient light
3. Add more light sources
4. Adjust camera distance

---

## Output Examples

### Typical Output Structure

```
visualizations/
├── gifs/
│   ├── S1_0.gif           # Rotating mesh
│   ├── S2_0.gif
│   └── ...
├── sdf/
│   ├── S1_sdf.png         # SDF cross-section
│   ├── S1_sdf_anim.gif    # Animated SDF
│   └── ...
├── colored/
│   ├── S1_0_colored.ply   # Error-colored PCD
│   └── ...
├── comparison/
│   ├── demo.png           # Multi-panel comparison
│   └── ...
└── axes/
    ├── S1_0_with_axes.ply # PCA axes visualization
    └── ...
```

---

## Next Step

Once visualizations are ready, proceed to:
→ **[Step 6: Automation](06_AUTOMATION.md)**

Or if you want to automate the entire pipeline, jump directly there!

---

## Related Files

- `create_promo_gif.py` - Create rotating mesh GIFs
- `visualize_sdf_simple.py` - Simple SDF viewer
- `visualize_sdf_gif.py` - Animated SDF viewer
- `visualize_colored_pointclouds.py` - Error visualization
- `demo_visualization.py` - Demo figure generation
- `generate_promo_gifs_test.sh` - Batch GIF generation

---

**Last Updated**: October 31, 2025

