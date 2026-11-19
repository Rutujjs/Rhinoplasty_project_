"""
BiSeNet Face Parsing & Nasal Segmentation Engine.

Implements real-time deep semantic face parsing using BiSeNet with ResNet-18 backbone
trained on CelebAMask-HQ (19 semantic classes).
Segments patient-specific nasal anatomy across all 9 canonical clinical viewpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard CelebAMask-HQ 19 class definitions
CELEBAMASK_HQ_CLASSES = [
    "background",   # 0
    "skin",         # 1
    "l_brow",       # 2
    "r_brow",       # 3
    "l_eye",        # 4
    "r_eye",        # 5
    "eye_g",        # 6 (eyeglasses)
    "l_ear",        # 7
    "r_ear",        # 8
    "ear_r",        # 9 (earrings)
    "nose",         # 10
    "mouth",        # 11
    "u_lip",        # 12
    "l_lip",        # 13
    "neck",         # 14
    "neck_l",       # 15 (necklace)
    "cloth",        # 16
    "hair",         # 17
    "hat",          # 18
]

NOSE_CLASS_ID = 10
INPUT_SIZE = (512, 512)

# ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# Default model weight locations
DEFAULT_WEIGHT_PATHS = [
    Path(__file__).resolve().parents[1] / "nasal-deformation-pipeline-main" / "models" / "bisenet" / "weights" / "resnet18.onnx",
    Path(__file__).resolve().parent / "weights" / "resnet18.onnx",
]


class BiSeNetSegmentor:
    """
    Patient-specific facial parsing segmentor using BiSeNet ResNet-18.
    """

    def __init__(self, weight_path: Optional[Union[str, Path]] = None) -> None:
        self.session: Any = None
        self.weight_path: Optional[Path] = None
        self._load_status: Dict[str, Any] = {
            "loaded": False,
            "backend": None,
            "weight_path": None,
            "num_classes": 19,
            "error": None,
        }
        self._initialize_session(weight_path)

    def _initialize_session(self, explicit_path: Optional[Union[str, Path]] = None) -> None:
        """Attempt to locate and load the ONNX model weights."""
        candidates = []
        if explicit_path:
            candidates.append(Path(explicit_path))
        candidates.extend(DEFAULT_WEIGHT_PATHS)

        chosen_path = None
        for cand in candidates:
            if cand.exists() and cand.is_file() and cand.stat().st_size > 10 * 1024 * 1024:
                chosen_path = cand
                break

        if not chosen_path:
            msg = f"BiSeNet ONNX weights not found in candidates: {[str(c) for c in candidates]}"
            logger.warning(msg)
            self._load_status = {
                "loaded": False,
                "backend": None,
                "weight_path": None,
                "num_classes": 19,
                "error": msg,
            }
            return

        try:
            import onnxruntime as ort

            # Prefer CPU provider for deterministic local execution
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(chosen_path), sess_options=opts, providers=["CPUExecutionProvider"])
            self.weight_path = chosen_path
            self._load_status = {
                "loaded": True,
                "backend": "onnxruntime",
                "weight_path": str(chosen_path),
                "num_classes": 19,
                "error": None,
            }
            logger.info("BiSeNet ResNet-18 model successfully loaded from %s", chosen_path)
        except Exception as exc:
            logger.exception("Failed to load BiSeNet session: %s", exc)
            self.session = None
            self._load_status = {
                "loaded": False,
                "backend": None,
                "weight_path": str(chosen_path),
                "num_classes": 19,
                "error": str(exc),
            }

    @property
    def is_available(self) -> bool:
        """Return whether actual model inference is ready and available."""
        return self.session is not None and self._load_status.get("loaded", False)

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic status of model loading."""
        return dict(self._load_status)

    def predict_full_parsing(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Run semantic segmentation inference on a BGR image.
        Returns full 19-class label map resized to original image dimensions.
        """
        if not self.is_available:
            raise RuntimeError(f"BiSeNet model is not loaded: {self._load_status.get('error')}")

        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel BGR numpy array.")

        h_orig, w_orig = image_bgr.shape[:2]

        # 1. Resize to model standard input resolution 512x512
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)

        # 2. Normalize with ImageNet mean/std
        norm = (img_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

        # 3. Transpose to NCHW tensor
        tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        # 4. Execute ONNX inference
        outputs = self.session.run(None, {"input": tensor})
        logits = outputs[0]  # shape: (1, 19, 512, 512)

        # 5. Argmax across class dimension
        labels_512 = np.argmax(logits[0], axis=0).astype(np.uint8)

        # 6. Resize back to original image dimensions
        labels_orig = cv2.resize(labels_512, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        return labels_orig

    def predict_nose_mask(
        self,
        image_bgr: np.ndarray,
        view_name: str = "frontal",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict patient-specific binary nasal mask for the given image and viewpoint.
        Returns (binary_mask, metadata).
        """
        from deformation.masking import clean_nose_mask

        h_orig, w_orig = image_bgr.shape[:2]

        if not self.is_available:
            raise RuntimeError(f"BiSeNet model is not loaded: {self._load_status.get('error')}")

        labels = self.predict_full_parsing(image_bgr)
        raw_nose = (labels == NOSE_CLASS_ID).astype(np.uint8) * 255

        # Check if nose was detected
        active_px = int(np.count_nonzero(raw_nose))
        if active_px == 0:
            logger.warning("BiSeNet predicted 0 nose pixels for view '%s'", view_name)
            raise ValueError(f"BiSeNet detected no nasal region in view '{view_name}'")

        cleaned_nose = clean_nose_mask(raw_nose, view_name)

        ys, xs = np.where(cleaned_nose > 0)
        metadata = {
            "segmentation_method": "BISENET_RESNET18_INFERENCE",
            "model_version": "bisenet_resnet18_celebamask",
            "model_weight_file": str(self.weight_path.name if self.weight_path else "unknown"),
            "viewpoint": view_name,
            "original_dimensions": [w_orig, h_orig],
            "nose_pixels": int(np.count_nonzero(cleaned_nose)),
            "nose_area_percentage": round(float(np.count_nonzero(cleaned_nose) / (w_orig * h_orig) * 100), 2),
            "bounding_box": {
                "xmin": int(xs.min()),
                "xmax": int(xs.max()),
                "ymin": int(ys.min()),
                "ymax": int(ys.max()),
            },
            "centroid": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
            "is_fallback": False,
        }

        return cleaned_nose, metadata


# Global singleton segmentor
bisenet_segmentor = BiSeNetSegmentor()

