"""
Pydantic schemas for the Kaggle Pipeline Test workflow.

These models define:
1. Input attributes for the test generation request.
2. Per-stage execution details for the 10 pipeline stages.
3. Status and preview responses returned to the frontend.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, model_validator


class StageStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class StageInfo(BaseModel):
    stage_id: int
    name: str
    status: StageStatus = StageStatus.PENDING
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    details: str | None = None
    error: str | None = None


class KaggleTestRequest(BaseModel):
    """
    Demographic and appearance configuration for a single test image generation.
    Enforces requested_count == 1 server-side.
    """
    population_group: str = Field(default="South Asian", description="Population / ethnic group")
    region: str = Field(default="India", description="Geographic region")
    age_min: int = Field(default=25, ge=0, le=120)
    age_max: int = Field(default=35, ge=0, le=120)
    sex_or_attribute: str = Field(default="female")
    skin_tone: str = Field(default="medium_brown")
    pose: str = Field(default="frontal")
    expression: str = Field(default="neutral")
    lighting: str = Field(default="neutral")
    medical_condition: str | None = Field(default=None)
    medical_severity: str | None = Field(default=None)
    resolution: int = Field(default=512, ge=256, le=1024)
    seed: int | None = Field(default=None, description="Optional deterministic seed")
    prompt_mode: str = Field(default="auto", description="Prompt mode: 'auto' or 'custom'")
    custom_prompt: str | None = Field(default=None, description="Custom prompt when prompt_mode is 'custom'")

    @model_validator(mode="after")
    def validate_prompt_mode(self) -> KaggleTestRequest:
        if self.prompt_mode not in ("auto", "custom"):
            raise ValueError(f"prompt_mode must be 'auto' or 'custom', got {self.prompt_mode!r}")
        if self.prompt_mode == "custom":
            if not self.custom_prompt or not self.custom_prompt.strip():
                raise ValueError("custom_prompt is required and cannot be empty when prompt_mode is 'custom'")
        return self


class PromptPreviewRequest(BaseModel):
    population_group: str = "South Asian"
    region: str = "India"
    age_min: int = 25
    age_max: int = 35
    sex_or_attribute: str = "female"
    skin_tone: str = "medium_brown"
    pose: str = "frontal"
    expression: str = "neutral"
    lighting: str = "neutral"
    medical_condition: str | None = None
    medical_severity: str | None = None


class PromptPreviewResponse(BaseModel):
    prompt: str


class KaggleTestRunSummary(BaseModel):
    run_id: str
    status: str
    current_stage_id: int
    stages: list[StageInfo]
    prompt: str
    seed: int
    provider_run_id: str | None = None
    image_url: str | None = None
    image_data_uri: str | None = None
    image_metadata: dict[str, Any] | None = None
    timing: dict[str, Any] = Field(default_factory=dict)
    validation_result: dict[str, Any] | None = None
    cleanup_result: dict[str, Any] | None = None
    error_summary: str | None = None
