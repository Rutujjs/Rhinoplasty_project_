from datetime import datetime

from pydantic import BaseModel, ConfigDict


class GeneratedImageResponse(BaseModel):
    id: int
    dataset_id: int
    image_path: str
    generation_model: str
    seed: int
    prompt: str
    status: str
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )