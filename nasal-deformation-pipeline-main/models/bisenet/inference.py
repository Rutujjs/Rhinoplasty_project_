"""
Inference module for BiSeNet in nasal-deformation-pipeline.
"""

from pathlib import Path
import sys

# Ensure deformation package is available
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deformation.bisenet_inference import BiSeNetSegmentor, bisenet_segmentor, NOSE_CLASS_ID

def load_model(model_name="resnet18", num_classes=19, weight_path=None, device="cpu"):
    """Compatibility loader matching legacy generate_masks interface."""
    return bisenet_segmentor

def prepare_image(image_rgb, input_size=(512, 512)):
    """Preprocess image for BiSeNet."""
    import cv2
    import numpy as np
    
    img_bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
    return img_bgr

