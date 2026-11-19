from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetCreate(BaseModel):
    dataset_name: str = Field(min_length=1, max_length=255)
    region: str = Field(min_length=1, max_length=100)
    population_group: str = Field(min_length=1, max_length=100)

    age_min: int = Field(ge=0, le=120)
    age_max: int = Field(ge=0, le=120)

    sex_or_attribute: str = Field(min_length=1, max_length=50)
    skin_tone: str = Field(min_length=1, max_length=50)
    pose: str = Field(min_length=1, max_length=50)
    expression: str = Field(min_length=1, max_length=50)

    medical_condition: str | None = Field(
        default=None,
        max_length=100
    )

    medical_severity: str | None = Field(
        default=None,
        max_length=50
    )

    num_images: int = Field(gt=0, default=9)
    resolution: int = Field(gt=0, default=512)

    # Multi-view patient ID and prompt overrides
    patient_id: str | None = Field(default=None, max_length=50)
    custom_prompt: str | None = Field(default=None, max_length=4000)
    viewpoint_prompts: str | None = Field(default=None, max_length=8000)

    # Nasal morphology deformation parameters
    morphology_type: str = Field(default="saddle", max_length=50)
    deformation_strength: float = Field(default=0.6, ge=0.0, le=2.0)
    deviated_direction: str = Field(default="right", max_length=20)

    @model_validator(mode="after")
    def validate_age_range(self):
        if self.age_min > self.age_max:
            raise ValueError(
                "age_min must be less than or equal to age_max"
            )
        return self


class DatasetResponse(BaseModel):
    id: int
    dataset_name: str
    region: str
    population_group: str

    age_min: int
    age_max: int

    sex_or_attribute: str
    skin_tone: str
    pose: str
    expression: str

    medical_condition: str | None
    medical_severity: str | None

    num_images: int
    resolution: int
    status: str

    generated_count: int
    generation_attempts: int

    patient_id: str | None = None
    custom_prompt: str | None = None
    viewpoint_prompts: str | None = None

    morphology_type: str = "saddle"
    deformation_strength: float = 0.6
    deviated_direction: str = "right"

    created_at: datetime
    timing: dict | None = None

    model_config = ConfigDict(
        from_attributes=True
    )
