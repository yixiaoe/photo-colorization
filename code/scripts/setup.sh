#!/bin/bash
# Environment verification — autoDL server (PyTorch 2.0 + CUDA 11.8)
set -e

echo "=== Python & PyTorch ==="
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo "=== torchvision Mask R-CNN ==="
python -c "from torchvision.models.detection import maskrcnn_resnet50_fpn; print('Mask R-CNN: OK')"

echo "=== Required packages ==="
python -c "import skimage, PIL, numpy; print('skimage/PIL/numpy: OK')"

echo "=== All checks passed ==="
