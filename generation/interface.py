from abc import ABC, abstractmethod

from backend.app.schemas.generation import (
    GenerationRequest,
    GeneratedImageResult,
)


class FaceGenerator(ABC):

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
    ) -> GeneratedImageResult:
        pass