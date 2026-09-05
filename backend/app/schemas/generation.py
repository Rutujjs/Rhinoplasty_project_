from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GenerationRequest(BaseModel):
    age_min: int = Field(ge=0, le=120)
    age_max: int = Field(ge=0, le=120)

    sex_or_attribute: str = Field(
        min_length=1,
        max_length=50
    )

    skin_tone: str = Field(
        min_length=1,
        max_length=50
    )

    population_group: str = Field(
        min_length=1,
        max_length=100
    )

    region: str = Field(
        min_length=1,
        max_length=100
    )

    pose: str = Field(
        min_length=1,
        max_length=50
    )

    expression: str = Field(
        min_length=1,
        max_length=50
    )

    medical_condition: str | None = Field(
        default=None,
        max_length=100
    )

    medical_severity: str | None = Field(
        default=None,
        max_length=50
    )

    resolution: int = Field(
        gt=0,
        le=2048
    )

    @model_validator(mode="after")
    def validate_age_range(self):
        if self.age_min > self.age_max:
            raise ValueError(
                "age_min must be less than or equal to age_max"
            )

        return self


class GeneratedImageResult(BaseModel):
    """
    Standard output returned by any face generation engine.
    """
    image_path: str
    generation_model: str
    seed: int
    prompt: str
    generation_time: float


class GeneratedImageResponse(BaseModel):
    id: int
    dataset_id: int
    image_url: str
    generation_model: str
    seed: int
    prompt: str
    status: str
    created_at: datetime
    model_config = ConfigDict(
        from_attributes=True
    )

class GenerationStatusResponse(BaseModel):
    dataset_id: int
    status: str

    requested_images: int
    generated_count: int
    generation_attempts: int

    progress_percentage: float