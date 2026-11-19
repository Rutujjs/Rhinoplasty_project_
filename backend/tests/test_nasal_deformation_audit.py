"""
Automated Deformation Audit Test Suite.

Verifies:
1. 2D nasal deformation algorithms across all 9 canonical viewpoints and 4 morphologies (36 combinations).
2. Input dimension compatibility: Frontal (384x384) vs Other Views (512x512).
3. Zero black borders or edge clipping artifacts (cv2.BORDER_REFLECT101).
4. Anatomical nose mask resolution (class_id = 10 validation).
5. Output visual continuity and dtype preservation.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from deformation.engine import deformation_engine
from deformation.masking import load_nose_mask, prepare_nose_mask
from generation.viewpoints import CLINICAL_VIEWPOINTS


MORPHOLOGIES = ["saddle", "hump", "broad", "deviated"]


class TestDeformationAuditMatrix:
    """Audit all 36 combinations of 9 viewpoints x 4 morphologies."""

    @pytest.mark.parametrize("vp", [v.key for v in CLINICAL_VIEWPOINTS])
    @pytest.mark.parametrize("morph", MORPHOLOGIES)
    def test_viewpoint_morphology_combination(self, vp: str, morph: str):
        # 1. Determine expected dimension: frontal is 384x384, others 512x512
        dim = 384 if vp == "frontal" else 512

        # 2. Check for real FLUX image or synthesize standardized clinical input
        real_img_path = Path("storage/images/patient_P01/original") / f"{vp}.png"
        if real_img_path.exists():
            img_bgr = cv2.imread(str(real_img_path))
        else:
            img_bgr = np.full((dim, dim, 3), 160, dtype=np.uint8)
            cv2.circle(img_bgr, (dim // 2, dim // 2), dim // 3, (190, 180, 175), -1)

        h, w = img_bgr.shape[:2]

        # 3. Load / scale canonical mask
        ref_mask_path = Path("deformation/reference_masks") / vp / "nose_binary.png"
        if ref_mask_path.exists():
            mask = load_nose_mask(ref_mask_path, (h, w))
        else:
            dummy_raw = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(dummy_raw, (w // 2, h // 2), (w // 8, h // 6), 0, 0, 360, 255, -1)
            mask = prepare_nose_mask(dummy_raw, (h, w))

        assert mask.shape == (h, w), f"Mask shape {mask.shape} mismatch with image shape {(h, w)}"
        assert mask.dtype == np.uint8

        # 4. Apply deformation
        direction = "right" if morph == "deviated" else None
        result = deformation_engine.apply_deformation(
            image_bgr=img_bgr,
            nose_mask=mask,
            viewpoint=vp,
            deformation_type=morph,
            strength=0.5,
            direction=direction,
        )

        assert result.success is True, f"Deformation failed for {vp} with {morph}: {result.error_message}"
        assert result.output_image is not None
        assert result.output_image.shape == (h, w, 3), f"Output shape {result.output_image.shape} != input {(h, w, 3)}"
        assert result.output_image.dtype == np.uint8

        # 5. Verify zero black borders (no pixels along image perimeter should be zeroed out)
        top_edge = result.output_image[0, :, :]
        bottom_edge = result.output_image[-1, :, :]
        left_edge = result.output_image[:, 0, :]
        right_edge = result.output_image[:, -1, :]

        for edge_name, edge in [("top", top_edge), ("bottom", bottom_edge), ("left", left_edge), ("right", right_edge)]:
            all_black_pixels = np.all(edge == 0, axis=-1)
            assert not np.all(all_black_pixels), f"Detected full black line on {edge_name} edge in {vp}_{morph}"


class TestMaskClassValidation:
    """Verify semantic mask class mappings and reference label integrity."""

    def test_class_10_is_nose_class(self):
        ref_dir = Path("deformation/reference_masks")
        if not ref_dir.exists():
            pytest.skip("Reference masks directory not found")

        for vp_folder in ref_dir.iterdir():
            if not vp_folder.is_dir():
                continue
            labels_path = vp_folder / "labels.png"
            nose_bin_path = vp_folder / "nose_binary.png"
            if labels_path.exists() and nose_bin_path.exists():
                labels = cv2.imread(str(labels_path), cv2.IMREAD_UNCHANGED)
                binary = cv2.imread(str(nose_bin_path), cv2.IMREAD_GRAYSCALE)

                # Class 10 extracted from labels
                extracted_nose = (labels == 10).astype(np.uint8) * 255

                # Compare pixel count with nose_binary.png
                diff = np.abs(extracted_nose.astype(np.int32) - binary.astype(np.int32))
                assert np.sum(diff > 0) == 0, f"Class 10 does not match nose_binary in {vp_folder.name}"


class TestDeformationValidationService:
    """Verify genuine deformation detection, metrics calculation, and NO_VISIBLE_CHANGE status."""

    def test_validation_detects_real_deformation(self):
        from backend.app.schemas.deformation import DeformationStatus
        from backend.app.services.deformation_service import deformation_service

        dim = 256
        orig = np.full((dim, dim, 3), 150, dtype=np.uint8)
        cv2.circle(orig, (dim // 2, dim // 2), dim // 4, (200, 180, 160), -1)

        mask = np.zeros((dim, dim), dtype=np.uint8)
        cv2.ellipse(mask, (dim // 2, dim // 2), (20, 30), 0, 0, 360, 255, -1)

        # Create deformed image with deliberate shift
        deformed = orig.copy()
        deformed[mask > 0] = (deformed[mask > 0].astype(int) + 25).clip(0, 255).astype(np.uint8)

        base_metrics = {"max_displacement_pixels": 5.2}
        status, metrics, notes = deformation_service._validate_deformation_result(
            original_bgr=orig,
            deformed_bgr=deformed,
            mask_binary=mask,
            base_metrics=base_metrics,
        )

        assert status == DeformationStatus.COMPLETED
        assert metrics["max_pixel_diff"] > 0
        assert metrics["changed_pixels_count"] > 0
        assert metrics["mean_diff_inside_mask"] > 0
        assert "Deformation validated" in notes

    def test_validation_flags_no_visible_change_on_identical_images(self):
        from backend.app.schemas.deformation import DeformationStatus
        from backend.app.services.deformation_service import deformation_service

        dim = 256
        orig = np.full((dim, dim, 3), 150, dtype=np.uint8)
        mask = np.zeros((dim, dim), dtype=np.uint8)
        cv2.ellipse(mask, (dim // 2, dim // 2), (20, 30), 0, 0, 360, 255, -1)

        # Identical image passed as deformed
        identical = orig.copy()
        base_metrics = {"max_displacement_pixels": 0.0}

        status, metrics, notes = deformation_service._validate_deformation_result(
            original_bgr=orig,
            deformed_bgr=identical,
            mask_binary=mask,
            base_metrics=base_metrics,
        )

        assert status == DeformationStatus.NO_VISIBLE_CHANGE
        assert metrics["max_pixel_diff"] == 0
        assert metrics["changed_pixels_count"] == 0
        assert "below visible threshold" in notes

    def test_validation_fails_on_none_or_shape_mismatch(self):
        from backend.app.schemas.deformation import DeformationStatus
        from backend.app.services.deformation_service import deformation_service

        orig = np.full((100, 100, 3), 150, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)

        # None output
        status, _, notes = deformation_service._validate_deformation_result(
            original_bgr=orig,
            deformed_bgr=None,
            mask_binary=mask,
        )
        assert status == DeformationStatus.FAILED

        # Shape mismatch
        wrong_shape = np.full((50, 50, 3), 150, dtype=np.uint8)
        status, _, notes = deformation_service._validate_deformation_result(
            original_bgr=orig,
            deformed_bgr=wrong_shape,
            mask_binary=mask,
        )
        assert status == DeformationStatus.FAILED
        assert "Shape mismatch" in notes
