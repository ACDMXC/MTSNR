from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from run_single_image import (
    compute_mean_profiles,
    compute_residual,
    find_single_tiff,
    save_comparison_pdf,
    save_convergence_pdf,
    save_mean_profiles_pdf,
)


class SingleImageRunnerTests(unittest.TestCase):
    def test_find_single_tiff_requires_exactly_one_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with self.assertRaisesRegex(ValueError, "exactly one TIFF"):
                find_single_tiff(folder)

            expected = folder / "example.TIF"
            tifffile.imwrite(expected, np.zeros((8, 9), dtype=np.uint16))
            self.assertEqual(find_single_tiff(folder), expected)

            tifffile.imwrite(folder / "second.tiff", np.zeros((8, 9), dtype=np.uint16))
            with self.assertRaisesRegex(ValueError, "exactly one TIFF"):
                find_single_tiff(folder)

    def test_residual_is_noisy_minus_proposed(self) -> None:
        noisy = np.array([[4.0, 2.0], [7.0, 1.0]], dtype=np.float32)
        proposed = np.array([[1.5, 2.5], [3.0, -1.0]], dtype=np.float32)
        expected = np.array([[2.5, -0.5], [4.0, 2.0]], dtype=np.float32)
        np.testing.assert_array_equal(compute_residual(noisy, proposed), expected)

    def test_mean_profiles_follow_row_and_column_axes(self) -> None:
        image = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        row_mean, column_mean = compute_mean_profiles(image)
        np.testing.assert_array_equal(row_mean, np.array([2.0, 5.0], dtype=np.float32))
        np.testing.assert_array_equal(column_mean, np.array([2.5, 3.5, 4.5], dtype=np.float32))

    def test_save_comparison_pdf_writes_a_pdf(self) -> None:
        noisy = np.arange(64, dtype=np.float32).reshape(8, 8)
        proposed = noisy * 0.9
        residual = compute_residual(noisy, proposed)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "comparison.pdf"
            save_comparison_pdf(noisy, proposed, residual, output, show=False)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)
            self.assertEqual(output.read_bytes()[:4], b"%PDF")

    def test_additional_diagnostic_pdf_writers(self) -> None:
        noisy = np.arange(64, dtype=np.float32).reshape(8, 8)
        proposed = noisy * 0.9
        history = {
            "normalized_primal_residual": [4.0e-4, 8.0e-5, 2.0e-5],
            "normalized_dual_residual": [6.0e-6, 1.0e-6, 3.0e-7],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            convergence = save_convergence_pdf(history, root / "convergence.pdf", show=False)
            profiles = save_mean_profiles_pdf(noisy, proposed, root / "profiles.pdf", show=False)
            for output in (convergence, profiles):
                self.assertTrue(output.is_file())
                self.assertGreater(output.stat().st_size, 1000)
                self.assertEqual(output.read_bytes()[:4], b"%PDF")


if __name__ == "__main__":
    unittest.main()
