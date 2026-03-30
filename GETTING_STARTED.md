# Getting Started

## Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- CUDA-capable GPU (optional, but recommended for faster inference)

## Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/AbdoSaba/Floorplan223D.git
cd Floorplan223D
pip install -r requirements.txt
```

### 2. Download Model Weights
Download `best.onnx` from the [Releases](../../releases) page and place it in the `weights/` folder:
```
weights/
└── best.onnx
```

### 3. Run the Application
```bash
python main.py
```

## Usage

1. Provide path to your floorplan image
2. Configure parameters:
   - Real Width (default: 3000mm)
   - Wall Height (default: 450mm)
   - Wall Thickness (default: 50mm)
3. View the 3D visualization
4. Export as `output.obj`

## Supported Image Formats
- JPG/JPEG
- PNG
- BMP

## Common Issues

**"Model not found at weights/best.onnx"**
- Download the weights from Releases and ensure correct folder structure

**Low detection accuracy**
- Ensure good image quality
- Try adjusting `CONF_THRESH` in `main.py`

## Development

To retrain the model:
1. Set up your Roboflow API key
2. Open `scripts/train_yolovl.ipynb`
3. Run the notebook for training and ONNX export

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Citation

If you use this project, please cite:
```bibtex
@software{floorplan223d,
  author = {AbdoSaba},
  title = {Floorplan223D: 2D to 3D Floorplan Reconstruction with YOLOv8},
  year = {2025},
  url = {https://github.com/AbdoSaba/Floorplan223D}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact the maintainer.
