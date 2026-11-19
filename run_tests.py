"""
Local test runner and verification script.

Run this from the project root in your activated venv:

    python run_tests.py

OR directly invoke pytest:

    python -m pytest backend/tests/test_manifest_and_prompts.py -v
    python -m pytest backend/tests/ -v --tb=short

This script also applies the notebook patch before running tests,
then prints a human-readable summary of results.
"""
import subprocess
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent
VENV_PYTHON = Path(sys.executable)


def get_python():
    """Find the correct Python interpreter."""
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def patch_notebook():
    """Apply the notebook patch (idempotent)."""
    patch_script = ROOT / "kaggle_notebook_check" / "patch_notebook.py"
    if not patch_script.exists():
        print("[SKIP] Notebook patch script not found, skipping.")
        return
    result = subprocess.run(
        [get_python(), str(patch_script)],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"[WARNING] Notebook patch failed:\n{result.stderr}")


def run_tests(test_path: str, label: str):
    """Run a test module and return (passed, failed, errors, output)."""
    python = get_python()
    cmd = [
        python, "-m", "pytest",
        test_path,
        "-v", "--tb=short", "--no-header",
        "-p", "no:warnings",
    ]
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:2000])

    return result.returncode


def main():
    if "--diagnostic" in sys.argv:
        import subprocess
        python = get_python()
        cmd = [python, str(ROOT / "diagnostic_suite.py")] + [arg for arg in sys.argv[1:] if arg != "--diagnostic"]
        res = subprocess.run(cmd)
        return res.returncode

    print("=" * 60)
    print("Synthetic APAC Face Dataset Generator - Test Suite")
    print("=" * 60)

    # Safety check: ensure MOCK provider is active for offline test execution
    import os
    os.environ["GENERATION_PROVIDER"] = "MOCK"
    os.environ["GENERATION_MODEL"] = "MOCK"
    os.environ["RUN_EMBEDDED_WORKER"] = "false"
    from backend.app.core.config import settings  # type: ignore
    provider = settings.effective_provider
    if provider != "MOCK":
        print(f"\n[SAFETY ABORT] effective_provider={provider!r} — must be 'MOCK'")
        print("Set GENERATION_MODEL=MOCK (or GENERATION_PROVIDER=MOCK) in backend/.env")
        sys.exit(1)
    print(f"\n[SAFETY OK] effective_provider={provider!r} (no remote Kaggle jobs)")

    # Apply notebook patch
    patch_notebook()

    # Run test suites
    suites = [
        ("backend/tests/test_manifest_and_prompts.py", "Phase B/G: Manifest + Prompt Tests"),
        ("backend/tests/test_kaggle_pipeline_test.py", "Phase B/E: Kaggle Pipeline Test UI & Service (NEW)"),
        ("backend/tests/test_generation.py", "Phase C: Generation API Tests"),
        ("backend/tests/test_kaggle_provider.py", "Phase D: Kaggle Provider Tests"),
        ("backend/tests/test_multiview.py", "Phase M: 9-Viewpoint Architecture Tests"),
        ("backend/tests/test_nasal_deformation.py", "Phase N: 2D Nasal Morphology Deformation Tests"),
        ("backend/tests/test_dataset_isolation.py", "Phase I: Multi-Dataset Storage & Data Isolation Tests"),
    ]

    results = {}
    for path, label in suites:
        rc = run_tests(path, label)
        results[label] = "PASSED" if rc == 0 else "FAILED"

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for label, status in results.items():
        marker = "+" if status == "PASSED" else "X"
        print(f"  {marker} [{status}] {label}")

    all_passed = all(v == "PASSED" for v in results.values())
    print("\nOverall:", "ALL PASSED" if all_passed else "SOME FAILURES")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
