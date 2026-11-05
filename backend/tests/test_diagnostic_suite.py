"""
Unit tests for the Kaggle Pipeline Diagnostic Suite and its Phase 5 read-only remote checks.

Verifies:
1. When --confirm-read-only-remote is NOT provided:
   - Remote checks (H01, H02, H03) are marked NOT RUN.
   - Zero network calls to Kaggle API are made.
   - All local checks (A-G) execute and pass.
2. When --confirm-read-only-remote IS provided:
   - Remote checks H01, H02, H03 execute with check_type="REMOTE".
   - Only read-only methods (get_config_value, dataset_status, kernels_status) are called.
   - Zero mutation methods (dataset_create_version, kernels_push, etc.) are called.
   - Remote successes are counted in remote_passed_count.
3. When remote checks fail (e.g., auth or network error):
   - Local checks remain completely intact and reported accurately.
   - Exceptions are safely caught and secrets/tokens are redacted.
4. Secret redaction helper redacts arbitrary tokens, passwords, and sensitive settings.
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from diagnostic_suite import (
    KagglePipelineDiagnosticSuite,
    _safe_redact,
    format_terminal_report,
)


def test_diagnostic_suite_without_remote_opt_in():
    """Verify that without remote opt-in, H01-H03 are NOT RUN and Kaggle API is never called."""
    with patch("diagnostic_suite._get_kaggle_api") as mock_get_api:
        suite = KagglePipelineDiagnosticSuite(include_remote_read_only=False)
        report = suite.execute_all()

        # Kaggle API must never be invoked
        mock_get_api.assert_not_called()

        # Check counts
        assert report.remote_not_run_count == 3
        assert report.remote_passed_count == 0
        assert report.remote_failed_count == 0
        assert report.local_passed_count == 31
        assert report.local_failed_count == 0

        # Verify check details for H01, H02, H03
        h_checks = [c for c in report.checks if c.category.startswith("H")]
        assert len(h_checks) == 3
        for c in h_checks:
            assert c.status == "NOT RUN"
            assert c.check_type == "REMOTE"
            assert "explicit opt-in flag" in c.details


def test_diagnostic_suite_with_remote_opt_in_success():
    """Verify that with remote opt-in, H01-H03 execute read-only queries with mocked KaggleApi."""
    mock_api = MagicMock()
    mock_api.CONFIG_NAME_USER = "username"
    mock_api.get_config_value.return_value = "apac_researcher"
    mock_api.dataset_status.return_value = "ready"
    mock_api.kernels_status.return_value = {"status": "complete"}

    with patch("diagnostic_suite._get_kaggle_api", return_value=mock_api):
        suite = KagglePipelineDiagnosticSuite(include_remote_read_only=True)
        report = suite.execute_all()

        # Counts
        assert report.remote_not_run_count == 0
        assert report.remote_passed_count == 3
        assert report.remote_failed_count == 0
        assert report.local_passed_count == 31
        assert report.failed_count == 0

        # Verify H01, H02, H03 status
        h_checks = {c.check_id: c for c in report.checks if c.category.startswith("H")}
        assert "H01" in h_checks and h_checks["H01"].status == "PASS" and h_checks["H01"].check_type == "REMOTE"
        assert "H02" in h_checks and h_checks["H02"].status == "PASS" and h_checks["H02"].check_type == "REMOTE"
        assert "H03" in h_checks and h_checks["H03"].status == "PASS" and h_checks["H03"].check_type == "REMOTE"

        # Verify ONLY read-only methods were invoked
        mock_api.get_config_value.assert_called()
        mock_api.dataset_status.assert_called_once()
        mock_api.kernels_status.assert_called_once()

        # Strict safety check: NO write methods may ever be called
        assert not hasattr(mock_api, "dataset_create_version") or mock_api.dataset_create_version.call_count == 0
        assert not hasattr(mock_api, "kernels_push") or mock_api.kernels_push.call_count == 0


def test_diagnostic_suite_remote_failure_isolates_local_checks():
    """Verify that if remote checks fail, local checks still pass and tokens are redacted."""
    sensitive_token = "4a5b6c7d8e9f0123456789abcdef0123"
    error_msg = f"Failed to authenticate with token={sensitive_token} connection timed out"

    mock_api = MagicMock()
    mock_api.get_config_value.side_effect = RuntimeError(error_msg)
    mock_api.dataset_status.side_effect = RuntimeError(f"Dataset unreachable: key={sensitive_token}")
    mock_api.kernels_status.side_effect = RuntimeError(f"Kernel query failed: password={sensitive_token}")

    with patch("diagnostic_suite._get_kaggle_api", return_value=mock_api):
        suite = KagglePipelineDiagnosticSuite(include_remote_read_only=True)
        report = suite.execute_all()

        # Local checks are completely unaffected
        assert report.local_passed_count == 31
        assert report.local_failed_count == 0

        # Remote checks failed
        assert report.remote_failed_count == 3
        assert report.remote_passed_count == 0

        # Ensure secrets were redacted in error summaries
        h_checks = [c for c in report.checks if c.category.startswith("H")]
        for c in h_checks:
            assert c.status == "FAIL"
            assert sensitive_token not in (c.error_summary or "")
            assert "[REDACTED" in (c.error_summary or "")


def test_safe_redact_helper():
    """Test secret redaction patterns."""
    assert _safe_redact("normal log message") == "normal log message"
    assert "REDACTED" in _safe_redact("User key=my_super_secret_key_12345 failed")
    assert "REDACTED" in _safe_redact("Token 1234567890abcdef1234567890abcdef is expired")
    assert "REDACTED" in _safe_redact("authorization: Bearer xyz1234567890abcdef1234567890abcdef")


def test_format_terminal_report_distinguishes_local_and_remote():
    """Verify terminal report format displays check types and distinct summary lines."""
    suite = KagglePipelineDiagnosticSuite(include_remote_read_only=False)
    report = suite.execute_all()
    text = format_terminal_report(report)

    assert "Local Checks:" in text
    assert "Remote Checks:" in text
    assert "[LOCAL ]" in text
    assert "[REMOTE]" in text
    assert "ALL LOCAL CHECKS PASSED (Remote checks skipped - opt-in required)" in text
