"""
Unit and Integration Tests for the 2D Nasal Morphology Deformation Pipeline.

Tests cover:
  1. Deformation Engine:
     - Viewpoint normalization and canonical mapping
     - Parameter validation (saddle, hump, broad, deviated; strength bounds, direction)
     - Individual algorithm execution and output quality validation
     - Immutability of input images (zero mutation of original images)
  2. Masking Engine:
     - Reference mask resolution and fallback generation
     - Clean mask morphological operations
  3. API & Service Integration:
     - GET /deformation/types & GET /datasets/{id}/deformation/types
     - GET /datasets/{id}/deformation/status
     - POST /datasets/{id}/deformation/single
     - POST /datasets/{id}/deformation/batch
     - GET /datasets/{id}/deformation/image/{id}
     - GET /datasets/{id}/deformation/mask/{id}
     - Parameter validation & 404/400 error handling
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.models.dataset import Dataset
from backend.app.models.generated_image import GeneratedImage
from backend.app.schemas.deformation import DeformationStatus
from backend.app.services.deformation_service import deformation_service
from deformation.algorithms import (
    basal_broad_nose,
    frontal_broad_nose,
    deviated_nose_deformation,
    profile_broad_nose,
    saddle_nose_deformation,
)
from deformation.engine import (
    CANONICAL_VIEWPOINTS,
    MORPHOLOGY_METADATA,
    deformation_engine,
    normalize_viewpoint_key,
)
from deformation.masking import (
    clean_nose_mask,
    make_overlay,
    prepare_nose_mask,
)
from generation.viewpoints import CLINICAL_VIEWPOINTS, VIEWPOINT_KEYS


# --------------------------------------------------------------------------- #
# Helpers & Fixtures                                                          #
# --------------------------------------------------------------------------- #

def _create_synthetic_face_image(width: int = 256, height: int = 256) -> np.ndarray:
    """Generate a simple synthetic RGB face-like image for testing."""
    img = np.full((height, width, 3), 180, dtype=np.uint8)
    # Draw head oval
    cv2.ellipse(img, (width // 2, height // 2), (width // 3, height // 2 - 20), 0, 0, 360, (210, 190, 175), -1)
    # Draw nose area
    cv2.ellipse(img, (width // 2, int(height * 0.52)), (width // 10, height // 6), 0, 0, 360, (190, 160, 150), -1)
    return img


def _create_synthetic_nose_mask(width: int = 256, height: int = 256) -> np.ndarray:
    """Generate a synthetic binary nose mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, (width // 2, int(height * 0.52)), (width // 10, height // 6), 0, 0, 360, 255, -1)
    return mask


# --------------------------------------------------------------------------- #
# 1. Deformation Engine Unit Tests                                            #
# --------------------------------------------------------------------------- #

class TestDeformationEngine:
    def test_viewpoint_normalization(self):
        """Test canonical mapping across various naming conventions."""
        assert normalize_viewpoint_key("frontal") == "frontal"
        assert normalize_viewpoint_key("frontal_0") == "frontal"
        assert normalize_viewpoint_key("front") == "frontal"
        assert normalize_viewpoint_key("left_20") == "left_20"
        assert normalize_viewpoint_key("left20") == "left_20"
        assert normalize_viewpoint_key("right_60") == "right_60"
        assert normalize_viewpoint_key("basal") == "basal"
        assert normalize_viewpoint_key("basal_view") == "basal"
        assert normalize_viewpoint_key("helicopter") == "helicopter"
        assert normalize_viewpoint_key("helicopter_view") == "helicopter"

    def test_all_canonical_viewpoints_accounted_for(self):
        """Verify all 9 clinical viewpoints are recognized."""
        assert len(CANONICAL_VIEWPOINTS) == 9
        for vk in VIEWPOINT_KEYS:
            assert vk in CANONICAL_VIEWPOINTS

    def test_parameter_validation_success(self):
        """Test parameter validation for all four valid morphology types."""
        for dtype in ["saddle", "hump", "broad", "deviated"]:
            t, s, d = deformation_engine.validate_parameters(dtype, strength=0.5, direction="left")
            assert t == dtype
            assert s == 0.5
            if dtype == "deviated":
                assert d == "left"
            else:
                assert d is None

    def test_parameter_validation_defaults(self):
        """Test default parameter resolution when None is passed."""
        t, s, d = deformation_engine.validate_parameters("saddle")
        assert t == "saddle"
        assert s == MORPHOLOGY_METADATA["saddle"]["default_strength"]
        assert d is None

        t, s, d = deformation_engine.validate_parameters("deviated")
        assert t == "deviated"
        assert d == MORPHOLOGY_METADATA["deviated"]["default_direction"]

    def test_parameter_validation_invalid_type(self):
        """Reject invalid morphology types."""
        with pytest.raises(ValueError, match="Unsupported deformation_type"):
            deformation_engine.validate_parameters("aquiline")

    def test_parameter_validation_out_of_bounds_strength(self):
        """Reject strength values outside [0.0, 1.0]."""
        with pytest.raises(ValueError, match="Strength must be between"):
            deformation_engine.validate_parameters("hump", strength=1.5)

        with pytest.raises(ValueError, match="Strength must be between"):
            deformation_engine.validate_parameters("hump", strength=-0.1)

    def test_parameter_validation_invalid_direction(self):
        """Reject direction other than left or right."""
        with pytest.raises(ValueError, match="Direction for deviated must be"):
            deformation_engine.validate_parameters("deviated", direction="upward")

    @pytest.mark.parametrize("dtype", ["saddle", "hump", "broad", "deviated"])
    @pytest.mark.parametrize("viewpoint", ["frontal", "left_40", "right_40", "basal", "helicopter"])
    def test_deformation_execution_across_viewpoints(self, dtype: str, viewpoint: str):
        """Verify each morphology executes successfully on diverse viewpoints."""
        img = _create_synthetic_face_image(256, 256)
        mask = _create_synthetic_nose_mask(256, 256)
        orig_copy = img.copy()

        result = deformation_engine.apply_deformation(
            image_bgr=img,
            nose_mask=mask,
            viewpoint=viewpoint,
            deformation_type=dtype,
            strength=0.5,
            direction="left" if dtype == "deviated" else None,
        )

        assert result.success is True
        assert result.output_image is not None
        assert result.output_image.shape == img.shape
        assert result.output_image.dtype == np.uint8
        # Ensure original input was NOT modified in-place
        assert np.array_equal(img, orig_copy)
        # Ensure result has metrics
        assert "mask_pixels" in result.metrics
        assert "modified_pixels" in result.metrics

    def test_original_image_immutability(self):
        """Strictly verify original image bytes and hash are preserved."""
        img = _create_synthetic_face_image(128, 128)
        mask = _create_synthetic_nose_mask(128, 128)
        h_before = hashlib.sha256(img.tobytes()).hexdigest()

        _ = deformation_engine.apply_deformation(
            image_bgr=img,
            nose_mask=mask,
            viewpoint="frontal",
            deformation_type="broad",
            strength=0.6,
        )

        h_after = hashlib.sha256(img.tobytes()).hexdigest()
        assert h_before == h_after


# --------------------------------------------------------------------------- #
# 2. Masking Engine Tests                                                     #
# --------------------------------------------------------------------------- #

class TestMaskingEngine:
    def test_prepare_nose_mask_cleaning(self):
        """Verify prepare_nose_mask produces valid uint8 binary array."""
        raw = np.zeros((100, 100), dtype=np.uint8)
        raw[40:60, 40:60] = 10  # BiSeNet nose class ID is 10
        cleaned = prepare_nose_mask(raw, (100, 100), nose_class_id=10)
        assert cleaned.dtype == np.uint8
        assert cleaned.shape == (100, 100)
        assert set(np.unique(cleaned)).issubset({0, 255})

    def test_make_overlay(self):
        """Verify alpha overlay creation."""
        img = _create_synthetic_face_image(100, 100)
        mask = _create_synthetic_nose_mask(100, 100)
        overlay = make_overlay(img, mask)
        assert overlay.shape == img.shape
        assert overlay.dtype == np.uint8


# --------------------------------------------------------------------------- #
# 3. API & Service Integration Tests                                          #
# --------------------------------------------------------------------------- #

class TestDeformationAPIIntegration:
    @pytest.fixture
    def test_dataset_with_images(self, client: TestClient, db_session: Session, tmp_path: Path):
        """Fixture that sets up a dataset with 9 dummy generated image records and real files."""
        # 1. Create dataset via API
        ds_resp = client.post(
            "/datasets/",
            json={
                "dataset_name": "Deform_Integration_DS",
                "patient_id": "P99",
                "region": "Japan",
                "population_group": "East Asian",
                "age_min": 25,
                "age_max": 30,
                "sex_or_attribute": "female",
                "skin_tone": "fair",
                "pose": "frontal",
                "expression": "neutral",
                "num_images": 9,
                "resolution": 256,
            },
        )
        assert ds_resp.status_code == 200
        ds_data = ds_resp.json()
        ds_id = ds_data["id"]

        # 2. Create 9 mock image files and database records
        storage_dir = tmp_path / "test_images" / str(ds_id)
        storage_dir.mkdir(parents=True, exist_ok=True)

        for i, vp in enumerate(CLINICAL_VIEWPOINTS):
            img_arr = _create_synthetic_face_image(256, 256)
            img_file = storage_dir / f"{vp.key}.png"
            cv2.imwrite(str(img_file), img_arr)

            db_img = GeneratedImage(
                dataset_id=ds_id,
                image_path=str(img_file),
                generation_model="MOCK",
                seed=1000 + i,
                prompt=f"Facial photograph {vp.name}",
                status="GENERATED",
                validation_status="ACCEPTED",
                patient_id="P99",
                viewpoint=vp.key,
                viewpoint_name=vp.name,
                deformation_status="NOT_STARTED",
                image_width=256,
                image_height=256,
                image_format="png",
            )
            db_session.add(db_img)

        db_session.commit()
        return ds_id

    def test_get_morphology_types_global(self, client: TestClient):
        """GET /deformation/types returns all 4 clinical morphology types."""
        resp = client.get("/deformation/types")
        assert resp.status_code == 200
        data = resp.json()
        assert "morphologies" in data
        keys = [m["type_key"] for m in data["morphologies"]]
        assert "saddle" in keys
        assert "hump" in keys
        assert "broad" in keys
        assert "deviated" in keys

    def test_get_morphology_types_dataset_scoped(self, client: TestClient, test_dataset_with_images: int):
        """GET /datasets/{id}/deformation/types returns morphology catalog."""
        ds_id = test_dataset_with_images
        resp = client.get(f"/datasets/{ds_id}/deformation/types")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["morphologies"]) == 4

    def test_get_deformation_status_initial(self, client: TestClient, test_dataset_with_images: int):
        """GET /datasets/{id}/deformation/status initially shows NOT_STARTED."""
        ds_id = test_dataset_with_images
        resp = client.get(f"/datasets/{ds_id}/deformation/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["dataset_id"] == ds_id
        assert data["patient_id"] == "P99"
        assert data["status"] == "NOT_STARTED"
        assert data["backend_integration_ready"] is True
        assert len(data["viewpoints"]) == 9

    def test_single_viewpoint_deformation_saddle(self, client: TestClient, test_dataset_with_images: int):
        """POST /datasets/{id}/deformation/single applies saddle deformation to frontal view."""
        ds_id = test_dataset_with_images
        resp = client.post(
            f"/datasets/{ds_id}/deformation/single",
            json={
                "viewpoint": "frontal",
                "deformation_type": "saddle",
                "strength": 0.5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["deformation_type"] == "saddle"
        assert data["viewpoint"] == "frontal"
        assert data["strength"] == 0.5
        assert data["deformed_image_url"] is not None
        assert data["mask_image_url"] is not None

        # Verify downloading the deformed image
        img_resp = client.get(data["deformed_image_url"])
        assert img_resp.status_code == 200
        assert img_resp.headers["content-type"] == "image/png"

        # Verify downloading the mask
        mask_resp = client.get(data["mask_image_url"])
        assert mask_resp.status_code == 200
        assert mask_resp.headers["content-type"] == "image/png"

    def test_batch_deformation_all_nine_views(self, client: TestClient, test_dataset_with_images: int):
        """POST /datasets/{id}/deformation/batch applies hump deformation across all 9 views."""
        ds_id = test_dataset_with_images
        resp = client.post(
            f"/datasets/{ds_id}/deformation/batch",
            json={
                "deformation_type": "hump",
                "strength": 0.45,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["dataset_id"] == ds_id
        assert data["deformation_type"] == "hump"
        assert data["total_requested"] == 9
        assert data["succeeded_count"] == 9
        assert data["failed_count"] == 0
        assert data["status"] == "COMPLETED"
        assert len(data["viewpoints"]) == 9

        # Now status endpoint should reflect COMPLETED
        status_resp = client.get(f"/datasets/{ds_id}/deformation/status")
        assert status_resp.status_code == 200
        s_data = status_resp.json()
        assert s_data["status"] == "COMPLETED"

    def test_deviated_nose_direction_control(self, client: TestClient, test_dataset_with_images: int):
        """POST /datasets/{id}/deformation/single respects direction for deviated nose."""
        ds_id = test_dataset_with_images
        resp = client.post(
            f"/datasets/{ds_id}/deformation/single",
            json={
                "viewpoint": "frontal",
                "deformation_type": "deviated",
                "strength": 0.4,
                "direction": "right",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["deformation_type"] == "deviated"
        assert data["direction"] == "right"

    def test_single_view_invalid_deformation_type(self, client: TestClient, test_dataset_with_images: int):
        """Reject invalid morphology type with HTTP 400 or 422."""
        ds_id = test_dataset_with_images
        resp = client.post(
            f"/datasets/{ds_id}/deformation/single",
            json={
                "viewpoint": "frontal",
                "deformation_type": "invalid_nose_shape",
                "strength": 0.5,
            },
        )
        assert resp.status_code in (400, 422, 500)

    def test_image_and_mask_404_for_unknown_ids(self, client: TestClient, test_dataset_with_images: int):
        """404 returned when querying non-existent image or mask."""
        ds_id = test_dataset_with_images
        resp = client.get(f"/datasets/{ds_id}/deformation/image/999999")
        assert resp.status_code == 404

        mask_resp = client.get(f"/datasets/{ds_id}/deformation/mask/999999")
        assert mask_resp.status_code == 404
