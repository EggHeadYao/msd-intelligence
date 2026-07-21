from __future__ import annotations

import unittest

from l1_stats import classify_validation


FULL_COUNTS = {
    "same_artist": 10_000,
    "same_release": 10_000,
    "random": 10_000,
}
PAIR_TYPES = ("same_artist", "same_release", "random")


class L1ValidationStatusTest(unittest.TestCase):
    def test_supported_formal_run_passes(self) -> None:
        self.assertEqual(
            classify_validation(FULL_COUNTS, PAIR_TYPES, 10_000, False, True),
            (True, "PASS"),
        )

    def test_unsupported_formal_run_fails(self) -> None:
        self.assertEqual(
            classify_validation(FULL_COUNTS, PAIR_TYPES, 10_000, False, False),
            (True, "FAIL"),
        )

    def test_partial_mode_never_becomes_formal(self) -> None:
        self.assertEqual(
            classify_validation(FULL_COUNTS, PAIR_TYPES, 10_000, True, True),
            (False, "SMOKE_PASS"),
        )

    def test_missing_pair_type_is_not_formal(self) -> None:
        incomplete = {"same_artist": 10_000, "same_release": 10_000}
        self.assertEqual(
            classify_validation(incomplete, PAIR_TYPES, 10_000, False, True),
            (False, "SMOKE_PASS"),
        )


if __name__ == "__main__":
    unittest.main()
