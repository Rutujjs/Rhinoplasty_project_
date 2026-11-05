import json
from pathlib import Path

folder = Path(__file__).parent
original_path = folder / "apac-face-generator-manifest-ready.ipynb"
output_path = folder / "apac-face-generator-manifest-ready.patched.ipynb"

# Load the original notebook. Never modify it directly.
with original_path.open("r", encoding="utf-8") as file:
    notebook = json.load(file)

# Find the manifest settings cell.
settings_cells = [
    cell for cell in notebook["cells"]
    if cell.get("cell_type") == "code"
    and "MANIFEST_SEARCH_ROOT" in "".join(cell.get("source", []))
]

if len(settings_cells) != 1:
    raise RuntimeError(
        f"Expected one settings cell; found {len(settings_cells)}. Nothing saved."
    )

settings_cell = settings_cells[0]
settings_source = "".join(settings_cell["source"])

old_steps = '        "num_inference_steps": int(GENERATION_MANIFEST.get("num_inference_steps", 12)),'
new_steps = '''        "num_inference_steps": int(
            GENERATION_MANIFEST.get("provider_config", {}).get(
                "num_inference_steps",
                GENERATION_MANIFEST.get("num_inference_steps", 12),
            )
        ),'''

old_guidance = '        "guidance_scale": float(GENERATION_MANIFEST.get("guidance_scale", 1.0)),'
new_guidance = '''        "guidance_scale": float(
            GENERATION_MANIFEST.get("provider_config", {}).get(
                "guidance_scale",
                GENERATION_MANIFEST.get("guidance_scale", 1.0),
            )
        ),'''

if settings_source.count(old_steps) != 1:
    raise RuntimeError("Inference-steps line did not match exactly. Nothing saved.")

if settings_source.count(old_guidance) != 1:
    raise RuntimeError("Guidance-scale line did not match exactly. Nothing saved.")

settings_source = settings_source.replace(old_steps, new_steps, 1)
settings_source = settings_source.replace(old_guidance, new_guidance, 1)
compile(settings_source, "<manifest-settings-cell>", "exec")
settings_cell["source"] = settings_source.splitlines(keepends=True)

# Find the main batch-generation cell.
batch_cells = [
    cell for cell in notebook["cells"]
    if cell.get("cell_type") == "code"
    and "summary_document =" in "".join(cell.get("source", []))
    and "batch_records" in "".join(cell.get("source", []))
]

if len(batch_cells) != 1:
    raise RuntimeError(
        f"Expected one batch summary cell; found {len(batch_cells)}. Nothing saved."
    )

batch_cell = batch_cells[0]
batch_source = "".join(batch_cell["source"])

old_summary_fields = '''            "filename": r["filename"],
            "seed": r["seed"],
            "status": r["status"],
            "path": r.get("path"),
            "error": r.get("error"),'''

new_summary_fields = '''            "filename": r["filename"],
            "seed": r["seed"],
            "prompt": r["prompt"],
            "width": r["width"],
            "height": r["height"],
            "status": r["status"],
            "path": r.get("path"),
            "error": r.get("error"),'''

if batch_source.count(old_summary_fields) != 1:
    raise RuntimeError(
        "Summary fields did not match exactly. Nothing saved."
    )

batch_source = batch_source.replace(
    old_summary_fields, new_summary_fields, 1
)
compile(batch_source, "<batch-generation-cell>", "exec")
batch_cell["source"] = batch_source.splitlines(keepends=True)

# Write the patched copy only.
with output_path.open("w", encoding="utf-8", newline="\n") as file:
    json.dump(notebook, file, ensure_ascii=True, indent=1)
    file.write("\n")

# Reopen and verify the output.
with output_path.open("r", encoding="utf-8") as file:
    verified = json.load(file)

verified_cells = [
    "".join(cell.get("source", []))
    for cell in verified["cells"]
    if cell.get("cell_type") == "code"
]

verified_settings = next(
    source for source in verified_cells
    if "MANIFEST_SEARCH_ROOT" in source
)
verified_batch = next(
    source for source in verified_cells
    if "summary_document =" in source and "batch_records" in source
)

assert '"provider_config"' in verified_settings
assert '"num_inference_steps"' in verified_settings
assert '"guidance_scale"' in verified_settings
assert '"prompt": r["prompt"]' in verified_batch
assert '"width": r["width"]' in verified_batch
assert '"height": r["height"]' in verified_batch

print("SUCCESS: Both fixes applied and verified.")
print("Original notebook: unchanged")
print("Backup notebook: unchanged")
print("Patched notebook:", output_path.name)