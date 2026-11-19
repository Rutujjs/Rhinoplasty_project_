import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

from generation.kaggle_provider import KaggleBatchProvider
from generation.manifest import GenerationManifest, ManifestProviderConfig

manifest = GenerationManifest(
    dataset_id=10,
    job_id=999,
    batch_id="P01_test_diagnostic",
    requested_count=1,
    image_width=512,
    image_height=512,
    population_group="East Asian",
    region="Korea",
    age_min=25,
    age_max=30,
    sex_or_attribute="female",
    skin_tone="fair",
    pose="frontal",
    expression="neutral",
    seeds=[12345],
    provider_config=ManifestProviderConfig(
        provider="KAGGLE_BATCH",
        output_format="png",
    ),
)

provider = KaggleBatchProvider()
print("Calling provider.prepare_batch(manifest)...")
res = provider.prepare_batch(manifest)
print(f"Outcome: {res.outcome}")
print(f"Succeeded: {res.succeeded}")
print(f"Message: {res.message}")
print(f"Error detail: {res.error_detail}")
