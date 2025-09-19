# Camera Pose Estimation System

A comprehensive computer vision system for estimating camera pose based on object recognition using multiple detection methods including corner detection, SIFT feature matching, and bounding box detection.

## Overview

This project implements a robust camera pose estimation system that can determine the 3D position and orientation of a camera relative to known objects in the scene. The system supports multiple detection algorithms and provides accurate pose estimation for various applications including robotics, augmented reality, and computer vision research.

## Features

- **Multiple Detection Methods**: Supports three different pose estimation approaches:
  - Corner detection (Method 0) - Custom and OpenCV implementations
  - SIFT feature matching (Method 1) - Scale-invariant feature transform
  - Bounding box detection (Method 2) - YOLO-based object detection
- **Camera Calibration**: Built-in camera calibration functionality with support for different lens configurations
- **RANSAC Integration**: Robust pose estimation with outlier rejection
- **Real-time Processing**: Optimized for performance with timing measurements
- **Flexible Object Support**: Pre-configured for cards, cups, and other objects with customizable 3D models

## Installation

### Prerequisites

- Python 3.7+
- OpenCV 4.4.0+
- NumPy 1.23.1+

### Required Libraries

```bash
pip install numpy==1.23.1
pip install opencv-contrib-python==4.4.0.42
pip install opencv-python==4.4.0.42
```

### Optional Dependencies (for YOLO detection)

```bash
pip install ultralytics
pip install torch
pip install torchvision
```

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ObjectLocation
   ```

2. **Run the main pose estimation script**
   ```bash
   python process_corner.py
   ```

3. **Switch between detection methods**
   ```python
   method = 0  # 0: Corner detection, 1: SIFT, 2: Bounding box
   ```

## Usage

### Basic Usage

The main entry point is `process_corner.py`. Configure your input paths:

```python
# Image and data paths
data_path = "origin_img/measureddata/img5.jpg"
template = cv2.imread('template/card1.jpg', 0)
label_path = "origin_img/measureddata/data/data5.txt"
```

### Detection Methods

#### Method 0: Corner Detection
- Uses custom corner detection algorithm or OpenCV's `goodFeaturesToTrack`
- Switch between implementations:
```python
# Custom implementation
corners = corner.my_good_featuressToTrack(edges, maxCorners=30, qualityLevel=0.01, minDistance=30, mask=mask)

# OpenCV implementation
corners = cv2.goodFeaturesToTrack(edges, maxCorners=30, qualityLevel=0.01, minDistance=30, mask=mask)
```

#### Method 1: SIFT Feature Matching
- Template-based matching using SIFT features
- Requires template images in `/template` directory
- Supports various object types (cards, cups, etc.)

#### Method 2: Bounding Box Detection
- YOLO-based object detection
- Enable by uncommenting YOLO code in `process_corner.py`:
```python
model = YOLO("models/best.pt")
results = model(data_path)
```

### Camera Calibration

Use `calibration.py` to calibrate your camera:

```python
# Run calibration with chessboard images
python calibration.py
```

Configure different camera intrinsics:
```python
# Example for 26mm lens
mtx = np.array([[751.73392551, 0, 753.85550879],
                [0, 753.8908959, 732.11882348],
                [0, 0, 1]])
```

## Project Structure

```
ObjectLocation/
├── process_corner.py      # Main pose estimation script
├── location.py            # Core pose calculation functions
├── calibration.py         # Camera calibration utilities
├── corner.py              # Custom corner detection
├── SIFT.py                # SIFT feature matching
├── predict.py             # YOLO prediction utilities
├── preprocess.py          # Image preprocessing
├── display.py             # Visualization utilities
├── origin_img/            # Test images and data
│   └── measureddata/      # Measured test data
├── template/              # Template images for SIFT
├── calibrate_data/        # Camera calibration data
└── models/                # Pre-trained models
```

## API Reference

### Core Functions

#### `calculate_relative_position(K, pts3d, pts2d)`
Calculates camera pose using PnP algorithm.
- **K**: Camera intrinsic matrix
- **pts3d**: 3D object points
- **pts2d**: Corresponding 2D image points
- **Returns**: Rotation matrix R and translation vector t

#### `calculate_relative_position_ransac(K, pts3d, pts2d)`
Robust pose estimation with RANSAC outlier rejection.
- **Returns**: R, t, and inlier indices

#### `get_camera_origin_in_world_coord(R_wc, t_wc)`
Converts camera pose to world coordinates.
- **Returns**: Camera origin in world coordinate system

## Testing

Test images are located in `/origin_img/measureddata/` with corresponding data files in `/origin_img/measureddata/data/`. The system includes test cases for different object distances and orientations.

## Results

The system provides visual output showing:
- Detected feature points
- Estimated camera position
- Processing time measurements
- Pose estimation accuracy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV community for computer vision tools
- Ultralytics for YOLO implementation
- Contributors to the computer vision research community

## Citation

If you use this project in your research, please cite:

```bibtex
@software{camera_pose_estimation,
  title={Camera Pose Estimation System},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-username]/ObjectLocation}
}
```

## Support

For questions and support, please open an issue on GitHub or contact the maintainers.
