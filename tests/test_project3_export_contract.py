"""Tests for the standalone Project 3 prediction-export contract."""

from __future__ import annotations

import unittest

from scripts.project3_export_contract import (
    forward_transform,
    inverse_transform,
    parse_ordinal_class,
)


class Project3ExportContractTests(unittest.TestCase):
    def test_parse_ordinal_class_preserves_full_six_band_space(self) -> None:
        self.assertEqual(parse_ordinal_class("FQ"), ("FQ", 0))
        self.assertEqual(parse_ordinal_class("A9.9"), ("A", 1))
        self.assertEqual(parse_ordinal_class("C1.2"), ("C", 3))
        self.assertEqual(parse_ordinal_class("X2.5"), ("X", 5))

    def test_parse_ordinal_class_rejects_malformed_labels(self) -> None:
        for label in ("", "C", "Q1.0", "FQ1", "M-1.0"):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    parse_ordinal_class(label)

    def test_transform_round_trips_for_all_completed_target_types(self) -> None:
        cases = (
            (1e-7, {"name": "log10_scaled", "scale": 1e-8}),
            (1e-7, {"name": "log10_1p_scaled", "scale": 1e-8}),
            (2.5, {"name": "log10_1p"}),
        )
        for raw, transform in cases:
            with self.subTest(transform=transform):
                self.assertAlmostEqual(
                    inverse_transform(forward_transform(raw, transform), transform), raw
                )


if __name__ == "__main__":
    unittest.main()
