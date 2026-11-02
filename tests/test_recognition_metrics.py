"""Hand-checked edit alignment, error-type rates and target-length buckets."""
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.recognition_metrics import ErrorTally, align_counts, bin_label, tally


class AlignmentTests(unittest.TestCase):
    def cases(self):
        return [
            ("HELLO", "HELLO", (0, 0, 0)),
            ("", "", (0, 0, 0)),
            ("A", "", (1, 0, 0)),
            ("", "A", (0, 0, 1)),
            ("HELLO", "HLLO", (1, 0, 0)),           # dropped character
            ("HELLO", "HELLLO", (0, 0, 1)),         # extra character
            ("HELLO", "HELLP", (0, 1, 0)),          # one wrong character
            ("LETTER", "LETR", (2, 0, 0)),          # collapsed doubled letter plus a vowel
            ("ABCD", "AB", (2, 0, 0)),
            ("AB", "ABCD", (0, 0, 2)),
            ("AB", "BA", (0, 2, 0)),                # equal cost ties resolve to substitutions
            ("BEST", "BEST ", (0, 0, 1)),           # a trailing space is an insertion
            ("12345", "12O45", (0, 1, 0)),
            ("AND", "AND AND", (0, 0, 4)),
        ]

    def test_hand_written_alignments(self):
        for expected, actual, wanted in self.cases():
            with self.subTest(expected=expected, actual=actual):
                self.assertEqual(align_counts(expected, actual), wanted)

    def test_totals_match_the_edit_distance_used_for_cer(self):
        from scripts.train_neural import edit_distance
        rng = random.Random(7)
        alphabet = "ABCD "
        for _ in range(400):
            expected = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
            actual = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
            deletions, substitutions, insertions = align_counts(expected, actual)
            with self.subTest(expected=expected, actual=actual):
                self.assertEqual(deletions + substitutions + insertions, edit_distance(expected, actual))
                self.assertLessEqual(deletions, len(expected))
                self.assertLessEqual(insertions, len(actual))
                self.assertEqual(len(expected) - deletions + insertions, len(actual))

    def test_transposition_is_not_one_operation(self):
        deletions, substitutions, insertions = align_counts("AB", "BA")
        self.assertEqual(deletions + substitutions + insertions, 2)


class SummaryTests(unittest.TestCase):
    def pairs(self):
        return [("CAT", "CAT"),           # length 3, exact
                ("DOG", "DG"),            # length 3, one deletion
                ("HORSE", "HORSES"),      # length 5, one insertion
                ("ZEBRA", "ZEBRB"),       # length 5, one substitution
                ("PARROT", "PARROTS")]    # length 6, one insertion

    def test_summary_adds_keys_without_changing_existing_ones(self):
        summary = tally(self.pairs())
        self.assertEqual(summary["clips"], 5)
        self.assertEqual(summary["characters"], 3 + 3 + 5 + 5 + 6)
        self.assertEqual(summary["errors"], 4)
        self.assertEqual(summary["exact"], 1)
        self.assertAlmostEqual(summary["cer"], 4 / 22)
        self.assertAlmostEqual(summary["letter_accuracy"], 1 - 4 / 22)
        self.assertAlmostEqual(summary["exact_sequence_accuracy"], 1 / 5)
        self.assertEqual((summary["deletions"], summary["substitutions"], summary["insertions"]), (1, 1, 2))
        self.assertAlmostEqual(summary["deletion_rate"], 1 / 22)
        self.assertAlmostEqual(summary["substitution_rate"], 1 / 22)
        self.assertAlmostEqual(summary["insertion_rate"], 2 / 22)

    def test_error_rates_sum_to_the_edit_distance(self):
        summary = tally(self.pairs())
        self.assertAlmostEqual(summary["deletion_rate"] + summary["substitution_rate"] + summary["insertion_rate"],
                               summary["cer"])

    def test_length_bins_follow_the_short_and_long_split(self):
        summary = tally(self.pairs())
        short, long = bin_label(1, 4), bin_label(5, None)
        self.assertEqual(summary["length_clips"], {short: 2, long: 3})
        self.assertEqual(summary["length_exact"], {short: 1, long: 0})
        self.assertAlmostEqual(summary["length_accuracy"][short], .5)
        self.assertEqual(summary["length_accuracy"][long], 0)
        self.assertAlmostEqual(summary["length_cer"][short], 1 / 6)
        self.assertAlmostEqual(summary["length_cer"][long], 3 / 16)

    def test_custom_bins_and_labels(self):
        summary = tally(self.pairs(), length_bins=((1, 2), (3, 4), (5, 6), (7, None)))
        self.assertEqual(summary["length_clips"], {"1-2": 0, "3-4": 2, "5-6": 3, "7+": 0})
        self.assertEqual(summary["length_accuracy"]["1-2"], None)
        self.assertEqual(summary["length_accuracy"]["3-4"], .5)

    def test_empty_group_does_not_divide_by_zero(self):
        summary = tally([])
        self.assertEqual(summary["clips"], 0)
        self.assertIsNone(summary["cer"])
        self.assertIsNone(summary["exact_sequence_accuracy"])
        self.assertEqual(summary["length_clips"], {"1-4": 0, "5+": 0})

    def test_tally_accumulates_incrementally(self):
        group = ErrorTally()
        for expected, actual in self.pairs():
            group.add(expected, actual)
        self.assertEqual(group.summary(), tally(self.pairs()))

    def test_merge_matches_incremental_accumulation(self):
        left, right = ErrorTally(), ErrorTally()
        for expected, actual in self.pairs()[:2]:
            left.add(expected, actual)
        for expected, actual in self.pairs()[2:]:
            right.add(expected, actual)
        merged = ErrorTally().merge(left).merge(right)
        self.assertEqual(merged.summary(), tally(self.pairs()))

    def test_merge_of_shared_pair_tally_counts_once_per_call(self):
        shared = ErrorTally()
        shared.add("DOG", "DG")
        group = ErrorTally()
        group.merge(shared)
        group.merge(shared)
        self.assertEqual(group.clips, 2)
        self.assertEqual(group.deletions, 2)
        self.assertEqual(group.summary()["length_clips"], {"1-4": 2, "5+": 0})

    def test_conservative_decoder_shows_up_as_deletions(self):
        summary = tally([("ADDRESS", "ADRESS"), ("STREET", "STRET"), ("NORTH", "NORH")])
        self.assertEqual(summary["deletions"], 3)
        self.assertEqual(summary["insertions"], 0)
        self.assertGreater(summary["deletion_rate"], summary["insertion_rate"])
        self.assertEqual(summary["length_cer"][bin_label(5, None)], 3 / 18)


if __name__ == "__main__":
    unittest.main()
