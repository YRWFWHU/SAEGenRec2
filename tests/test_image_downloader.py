"""Unit tests for image_downloader module."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


def test_extract_image_url_from_meta():
    """Test URL extraction from Amazon meta JSON."""
    from SAEGenRec.data_process.image_downloader import _extract_main_image_url

    meta = {"imageURLHighRes": ["https://example.com/img1.jpg", "https://example.com/img2.jpg"]}
    url = _extract_main_image_url(meta)
    assert url == "https://example.com/img1.jpg"


def test_extract_image_url_missing():
    """Test URL extraction when no image URLs."""
    from SAEGenRec.data_process.image_downloader import _extract_main_image_url

    meta = {"title": "Some item"}
    url = _extract_main_image_url(meta)
    assert url is None


def test_skip_existing_files():
    """Test that existing files are skipped."""
    from SAEGenRec.data_process.image_downloader import _should_skip

    with tempfile.TemporaryDirectory() as tmpdir:
        existing = os.path.join(tmpdir, "B001.jpg")
        with open(existing, "w") as f:
            f.write("fake image data")
        assert _should_skip(existing) is True
        assert _should_skip(os.path.join(tmpdir, "B002.jpg")) is False


def test_concurrency_parameter():
    """Test that concurrency parameter is respected."""
    from SAEGenRec.data_process.image_downloader import download_images

    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = os.path.join(tmpdir, "meta_Test.json")
        # Empty meta file — no downloads
        with open(meta_path, "w") as f:
            json.dump([], f)

        with patch("SAEGenRec.data_process.image_downloader.ThreadPoolExecutor") as mock_pool:
            mock_pool.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_pool.return_value.__exit__ = MagicMock(return_value=False)
            download_images(
                category="Test",
                data_dir=tmpdir,
                image_dir=os.path.join(tmpdir, "images"),
                concurrency=4,
            )
            mock_pool.assert_called_once_with(max_workers=4)


def test_zero_vector_fallback_for_missing_images():
    """Test zero-vector fallback is triggered for missing images."""
    from SAEGenRec.data_process.image_downloader import _download_one

    with tempfile.TemporaryDirectory() as tmpdir:
        result = _download_one("B001", None, tmpdir)
        assert result["status"] == "skipped_no_url"


def test_retry_on_network_failure():
    """Test retry logic on network errors."""
    from SAEGenRec.data_process.image_downloader import _download_with_retry

    with patch("requests.get") as mock_get:
        mock_get.side_effect = Exception("Connection error")
        success = _download_with_retry("https://example.com/img.jpg", "/tmp/test.jpg", retries=2)
        assert success is False
        assert mock_get.call_count == 2
