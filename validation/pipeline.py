"""
Image validation pipeline – baseline checks applied to every downloaded image.

Each check returns a CheckResult with one of three outcomes:
- PASSED         : check ran and the image passed
- FAILED         : check ran and the image failed (include reason)
- NOT_EVALUATED  : check was skipped (validator unavailable or disabled)

The pipeline collects all check results and determines the overall
validation outcome:
- If any check FAILED  →  image is REJECTED
- If all checks PASSED →  image is ACCEPTED
- If all checks are NOT_EVALUATED (no checks ran) →  NOT_EVALUATED

Extension points
----------------
Add a new check function with the signature:
    check_<name>(path: Path, context: ValidationContext) -> CheckResult

Then append it to BASELINE_CHECKS or register it via add_check().

Face detection, age estimation, medical-feature checks, and privacy
similarity screening are registered as optional checks that return
NOT_EVALUATED unless the required libraries/models are available.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Check result                                                                 #
# --------------------------------------------------------------------------- #

class CheckOutcome(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    NOT_EVALUATED = "NOT_EVALUATED"


@dataclass
class CheckResult:
    check_name: str
    outcome: CheckOutcome
    reason: str = ""

    @property
    def passed(self) -> bool:
        return self.outcome == CheckOutcome.PASSED

    @property
    def failed(self) -> bool:
        return self.outcome == CheckOutcome.FAILED


# --------------------------------------------------------------------------- #
# Validation context                                                           #
# --------------------------------------------------------------------------- #

@dataclass
class ValidationContext:
    """
    Extra information available to check functions.

    Fields are populated by the caller (e.g. the worker) before running
    the pipeline.  Checks should access only the fields they need and must
    not crash if optional fields are None.
    """
    expected_width: int | None = None
    expected_height: int | None = None
    allowed_formats: list[str] = field(default_factory=lambda: ["png", "jpg", "jpeg", "webp"])
    max_file_bytes: int = 20 * 1024 * 1024   # 20 MB
    job_id: int | None = None
    image_id_remote: str | None = None
    require_face_detection: bool = False
    enable_duplicate_detection: bool = True
    known_checksums: set[str] = field(default_factory=set)
    duplicate_hash_threshold: int = 10
    # Populated by the pipeline after opening the image:
    image_width: int | None = None
    image_height: int | None = None
    image_format: str | None = None
    file_size_bytes: int | None = None
    checksum_sha256: str | None = None


# --------------------------------------------------------------------------- #
# Type alias                                                                   #
# --------------------------------------------------------------------------- #

CheckFn = Callable[[Path, ValidationContext], CheckResult]


# --------------------------------------------------------------------------- #
# Baseline checks                                                              #
# --------------------------------------------------------------------------- #

def check_file_exists(path: Path, ctx: ValidationContext) -> CheckResult:
    """File must exist and be a regular file."""
    if not path.exists():
        return CheckResult("file_exists", CheckOutcome.FAILED, f"File not found: {path}")
    if not path.is_file():
        return CheckResult("file_exists", CheckOutcome.FAILED, f"Not a regular file: {path}")
    return CheckResult("file_exists", CheckOutcome.PASSED)


def check_file_size(path: Path, ctx: ValidationContext) -> CheckResult:
    """File size must be > 0 and ≤ max_file_bytes."""
    try:
        size = path.stat().st_size
        ctx.file_size_bytes = size
    except OSError as exc:
        return CheckResult("file_size", CheckOutcome.FAILED, str(exc))

    if size == 0:
        return CheckResult("file_size", CheckOutcome.FAILED, "File is empty")
    if size > ctx.max_file_bytes:
        return CheckResult(
            "file_size", CheckOutcome.FAILED,
            f"File size {size} exceeds limit {ctx.max_file_bytes}"
        )
    return CheckResult("file_size", CheckOutcome.PASSED)


def check_image_decodable(path: Path, ctx: ValidationContext) -> CheckResult:
    """File must be a valid image that can be opened by Pillow."""
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        return CheckResult("image_decodable", CheckOutcome.NOT_EVALUATED, "Pillow not installed")

    try:
        with Image.open(path) as img:
            img.verify()
        # Reopen after verify (verify() partially reads the stream).
        with Image.open(path) as img:
            img.load()
            ctx.image_width = img.width
            ctx.image_height = img.height
            ctx.image_format = (img.format or "").lower()
    except Exception as exc:
        return CheckResult("image_decodable", CheckOutcome.FAILED, f"Cannot decode image: {exc}")

    return CheckResult("image_decodable", CheckOutcome.PASSED)


def check_image_format(path: Path, ctx: ValidationContext) -> CheckResult:
    """File extension and detected format must be in the allowed list."""
    extension = path.suffix.lstrip(".").lower()
    detected = ctx.image_format or extension

    if detected not in ctx.allowed_formats:
        return CheckResult(
            "image_format", CheckOutcome.FAILED,
            f"Format '{detected}' not in allowed formats: {ctx.allowed_formats}"
        )
    return CheckResult("image_format", CheckOutcome.PASSED)


def check_image_dimensions(path: Path, ctx: ValidationContext) -> CheckResult:
    """If expected dimensions are configured, the image must match."""
    if ctx.expected_width is None or ctx.expected_height is None:
        return CheckResult(
            "image_dimensions", CheckOutcome.NOT_EVALUATED,
            "No expected dimensions configured"
        )
    if ctx.image_width is None or ctx.image_height is None:
        return CheckResult(
            "image_dimensions", CheckOutcome.NOT_EVALUATED,
            "Image dimensions not yet populated (run check_image_decodable first)"
        )

    if ctx.image_width != ctx.expected_width or ctx.image_height != ctx.expected_height:
        return CheckResult(
            "image_dimensions", CheckOutcome.FAILED,
            f"Expected {ctx.expected_width}×{ctx.expected_height}, "
            f"got {ctx.image_width}×{ctx.image_height}"
        )
    return CheckResult("image_dimensions", CheckOutcome.PASSED)


def check_checksum(path: Path, ctx: ValidationContext) -> CheckResult:
    """Compute SHA-256 and store it in context (always passes)."""
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ctx.checksum_sha256 = digest
    except OSError as exc:
        return CheckResult("checksum", CheckOutcome.FAILED, str(exc))
    return CheckResult("checksum", CheckOutcome.PASSED)


def check_duplicate(path: Path, ctx: ValidationContext) -> CheckResult:
    """
    Detect exact duplicates using SHA-256.

    Near-duplicate detection (perceptual hash) is available as an extension.
    NOT_EVALUATED is returned when duplicate detection is disabled.
    """
    if not ctx.enable_duplicate_detection:
        return CheckResult("duplicate", CheckOutcome.NOT_EVALUATED, "Duplicate detection disabled")

    if ctx.checksum_sha256 is None:
        return CheckResult(
            "duplicate", CheckOutcome.NOT_EVALUATED,
            "Checksum not available (run check_checksum first)"
        )

    if ctx.checksum_sha256 in ctx.known_checksums:
        return CheckResult(
            "duplicate", CheckOutcome.FAILED,
            f"Exact duplicate detected (sha256={ctx.checksum_sha256})"
        )

    ctx.known_checksums.add(ctx.checksum_sha256)
    return CheckResult("duplicate", CheckOutcome.PASSED)


# --------------------------------------------------------------------------- #
# Optional / extension checks                                                  #
# --------------------------------------------------------------------------- #

def check_face_detection(path: Path, ctx: ValidationContext) -> CheckResult:
    """
    Require at least one detectable face.

    Returns NOT_EVALUATED if:
    - Face detection is not required in settings
    - opencv-python is not installed

    Returns FAILED if no face is detected and detection is required.

    NOTE: Face detection is an optional feature.  A NOT_EVALUATED result is
    not treated as a failure.
    """
    if not ctx.require_face_detection:
        return CheckResult(
            "face_detection", CheckOutcome.NOT_EVALUATED,
            "Face detection not required"
        )

    try:
        import cv2
    except ImportError:
        return CheckResult(
            "face_detection", CheckOutcome.NOT_EVALUATED,
            "opencv-python not installed; face detection skipped"
        )

    try:
        img = cv2.imread(str(path))
        if img is None:
            return CheckResult("face_detection", CheckOutcome.FAILED, "Could not read image")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade_path = (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return CheckResult("face_detection", CheckOutcome.FAILED, "No face detected")

    except Exception as exc:
        return CheckResult(
            "face_detection", CheckOutcome.NOT_EVALUATED,
            f"Face detection error: {exc}"
        )

    return CheckResult("face_detection", CheckOutcome.PASSED)


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #

# Ordered list of baseline checks applied to every image.
BASELINE_CHECKS: list[CheckFn] = [
    check_file_exists,
    check_file_size,
    check_image_decodable,
    check_image_format,
    check_image_dimensions,
    check_checksum,
    check_duplicate,
    check_face_detection,
]


@dataclass
class ValidationResult:
    """
    Overall result of running the pipeline for a single image.
    """
    path: Path
    outcome: str            # ACCEPTED | REJECTED | NOT_EVALUATED
    check_results: list[CheckResult] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)

    # Metadata populated during the pipeline run.
    image_width: int | None = None
    image_height: int | None = None
    image_format: str | None = None
    file_size_bytes: int | None = None
    checksum_sha256: str | None = None

    @property
    def accepted(self) -> bool:
        return self.outcome == "ACCEPTED"

    @property
    def rejected(self) -> bool:
        return self.outcome == "REJECTED"

    def rejection_reasons_json(self) -> str:
        return json.dumps(self.rejection_reasons)


def validate_image(
    path: Path,
    context: ValidationContext,
    checks: list[CheckFn] | None = None,
) -> ValidationResult:
    """
    Run the validation pipeline on a single image file.

    Parameters
    ----------
    path    : absolute path to the image file
    context : validation context (expected dimensions, allowed formats, etc.)
    checks  : list of check functions to run; defaults to BASELINE_CHECKS

    Returns a ValidationResult with the aggregate outcome and all check results.
    """
    if checks is None:
        checks = BASELINE_CHECKS

    results: list[CheckResult] = []

    for check_fn in checks:
        try:
            result = check_fn(path, context)
            results.append(result)

            # Stop early on FAILED so we don't waste time on subsequent checks
            # for images that have already failed a critical check.
            if result.failed:
                if check_fn in (check_file_exists, check_image_decodable):
                    # Fatal checks – no point continuing.
                    break

        except Exception as exc:
            logger.exception("Check '%s' raised an exception for %s", check_fn.__name__, path)
            results.append(
                CheckResult(
                    check_fn.__name__,
                    CheckOutcome.NOT_EVALUATED,
                    f"Check raised an exception: {exc}",
                )
            )

    # Determine overall outcome.
    failed_checks = [r for r in results if r.failed]
    passed_checks = [r for r in results if r.passed]

    if failed_checks:
        overall = "REJECTED"
        reasons = [r.reason for r in failed_checks]
    elif passed_checks:
        overall = "ACCEPTED"
        reasons = []
    else:
        # All checks returned NOT_EVALUATED.
        overall = "NOT_EVALUATED"
        reasons = []

    return ValidationResult(
        path=path,
        outcome=overall,
        check_results=results,
        rejection_reasons=reasons,
        image_width=context.image_width,
        image_height=context.image_height,
        image_format=context.image_format,
        file_size_bytes=context.file_size_bytes,
        checksum_sha256=context.checksum_sha256,
    )
