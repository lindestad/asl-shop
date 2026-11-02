"""Tiny-corpus language model arithmetic and a hand-checked CTC prefix-beam lattice."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from collections import Counter
from itertools import product

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.char_lm import END, NEG, START, UNKNOWN, CharNgramLM, ctc_prefix_beam, rescore_nbest, transcripts_from_files

LABELS = ["", "A", "B"]  # index 0 is the CTC blank


def decode(alignment):
    text, previous = [], None
    for index in alignment:
        if index and index != previous:
            text.append(LABELS[index])
        previous = index
    return "".join(text)


def brute_force(log_probs):
    total = Counter()
    frames = len(log_probs)
    for alignment in product(range(len(LABELS)), repeat=frames):
        probability = 1.0
        for frame, index in enumerate(alignment):
            probability *= math.exp(log_probs[frame][index])
        total[decode(alignment)] += probability
    return total


class LanguageModelTests(unittest.TestCase):
    def model(self, texts, **kwargs):
        return CharNgramLM(alphabet="AB", **kwargs).fit(texts)

    def test_stupid_backoff_arithmetic(self):
        model = self.model(["AA"], order=3, backoff_weight=.4, unigram_pseudo=.1)
        self.assertEqual(model.trained_tokens, 3)
        self.assertEqual(sorted(model.lexicon), sorted({"A", "B", END, UNKNOWN}))
        self.assertAlmostEqual(model.probability(START, "A"), 1.0)            # seen once, no extra smoothing
        self.assertAlmostEqual(model.probability(START + "A", "A"), 1.0)
        self.assertAlmostEqual(model.probability("AA", END), 1.0)
        smoothed_unigram = (2 + .1) / (3 + .1 * 4)
        self.assertAlmostEqual(model.probability("B", "A"), .4 * smoothed_unigram)        # one backoff step
        self.assertAlmostEqual(model.probability("BB", "A"), .4 * .4 * smoothed_unigram)  # two steps
        self.assertAlmostEqual(model.probability("", "B"), .1 / (3 + .1 * 4))            # the floor
        self.assertAlmostEqual(model.extend_log_prob("ZQ", "B"), math.log(.4 * .4 * .1 / (3 + .1 * 4)))

    def test_zero_probability_never_survives(self):
        model = self.model(["AB"], order=4)
        for token in ("A", "B", UNKNOWN, END):
            for context in ("AB", "ZZ", "ABAB", ""):
                self.assertGreater(model.probability(context, token), 0)
                self.assertTrue(math.isfinite(model.log_prob(context, token)))

    def test_a_deterministic_corpus_has_perplexity_one(self):
        model = self.model(["A"], order=2)
        self.assertAlmostEqual(model.score("A"), 0.0)
        self.assertAlmostEqual(model.perplexity(["A"]), 1.0)
        self.assertGreater(model.perplexity(["BA"]), 1.0)

    def test_case_folding_and_unknown_characters(self):
        model = CharNgramLM(alphabet="BEST", order=3).fit(["best"])
        self.assertAlmostEqual(model.score("BEST"), model.score("best"))
        self.assertEqual(model.normalize("Q"), UNKNOWN)
        self.assertTrue(math.isfinite(model.score("Q")))
        self.assertLess(model.score("Q"), model.score("BEST"))

    def test_dev_perplexity_rises_on_unseen_text(self):
        model = CharNgramLM(order=4).fit(["aaa bbb", "aaa bbb ccc", "aaa bbb"])
        self.assertLess(model.perplexity(["aaa bbb"]), model.perplexity(["qz xk"]))

    def test_serialization_roundtrip_keeps_scores(self):
        model = CharNgramLM(alphabet="AB", order=3).fit(["AAAB"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lm.json.gz"
            model.save(path)
            loaded = CharNgramLM.load(path)
        self.assertEqual(loaded.order, model.order)
        self.assertEqual(loaded.totals["AA"], model.totals["AA"])
        self.assertAlmostEqual(loaded.score("AAAB"), model.score("AAAB"))
        self.assertAlmostEqual(loaded.extend_log_prob("AA", "B"), model.extend_log_prob("AA", "B"))

    def test_builder_takes_explicit_sources_and_skips_held_out_clips(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            packed = directory / "train_clips.json"
            packed.write_text(json.dumps([dict(name="train-a", text="ZEBRA CROSSING"),
                                          dict(name="test-z", text="XYLOPHONE QUARRY")]), encoding="utf-8")
            metadata = directory / "fsboard-metadata.json"
            metadata.write_text(json.dumps([dict(clipFilename="train-b_0_1.mp4", phrase="hancock park"),
                                            dict(clipFilename="test-z_0_1.mp4", phrase="XYLOPHONE QUARRY")]), encoding="utf-8")
            plain = directory / "extra.txt"
            plain.write_text("MAPLE STREET\n\n", encoding="utf-8")
            corpus = transcripts_from_files([packed, metadata, plain],
                                            exclude_names={"test-z", "test-z_0_1", "test-z_0"})
            self.assertEqual(corpus["texts"], ["ZEBRA CROSSING", "HANCOCK PARK", "MAPLE STREET"])
            self.assertEqual(corpus["excluded_records"], 2)
            self.assertEqual([row["records"] for row in corpus["sources"]], [1, 1, 1])
            model = CharNgramLM(order=5).fit(corpus["texts"])
        self.assertEqual(model.ngram_count("CROSS"), 1)    # trained 5-grams are present
        self.assertEqual(model.ngram_count("HANCO"), 1)
        self.assertEqual(model.ngram_count("QUARR"), 0)    # test-only text never reached the counts
        self.assertEqual(model.ngram_count("XYLOP"), 0)
        self.assertEqual(model.ngram_count("OPHON"), 0)
        self.assertAlmostEqual(model.probability("QUAR", "R"), model.probability("ZZZZ", "R"))

    def test_language_model_can_overrule_an_acoustically_better_hypothesis(self):
        model = CharNgramLM(alphabet="BESTQ").fit(["best"] * 20)
        hypotheses = [("BEST", -1.0), ("BESTQ", -0.2)]
        self.assertEqual(rescore_nbest(hypotheses, model, 0.0)[0]["text"], "BESTQ")
        ranked = rescore_nbest(hypotheses, model, 2.0)
        self.assertEqual(ranked[0]["text"], "BEST")
        language = {row["text"]: row["language"] for row in ranked}
        self.assertAlmostEqual(language["BEST"], 0.0)
        self.assertLess(language["BESTQ"], 0.0)
        self.assertAlmostEqual(ranked[0]["total"], ranked[0]["acoustic"] + 2 * ranked[0]["language"])


class CtcPrefixBeamTests(unittest.TestCase):
    def scores(self, results):
        return {text: (acoustic, language) for text, acoustic, language in results}

    def assert_matches_brute_force(self, log_probs):
        results = ctc_prefix_beam(np.asarray(log_probs, dtype=float), LABELS, beam_width=96,
                                  n_best=96, top_k=len(LABELS))
        expected = brute_force(log_probs)
        self.assertEqual({text for text, _, _ in results}, set(expected))
        for text, acoustic, _ in results:
            self.assertTrue(math.isclose(math.exp(acoustic), expected[text], rel_tol=1e-7, abs_tol=1e-12),
                            f"{text}: {math.exp(acoustic)} != {expected[text]}")
        self.assertEqual([text for text, _, _ in results], [text for text, _ in expected.most_common()])

    def test_two_frame_lattice_matches_all_alignments(self):
        rows = [[math.log(.4), math.log(.6), math.log(1e-9)], [math.log(.3), math.log(.5), math.log(.2)]]
        scores = self.scores(ctc_prefix_beam(np.asarray(rows), LABELS, beam_width=8, n_best=8, top_k=3))
        expected = brute_force(rows)
        self.assertEqual(set(scores), set(expected))
        for text, (acoustic, _) in scores.items():
            self.assertTrue(math.isclose(math.exp(acoustic), expected[text], rel_tol=1e-9))
        self.assertAlmostEqual(sum(math.exp(acoustic) for acoustic, _ in scores.values()), 1.0, places=6)

    def test_repeats_need_a_blank_and_are_not_double_counted(self):
        rows = [[NEG, math.log(.9), NEG], [NEG, math.log(.9), NEG]]  # A emitted twice in a row
        scores = self.scores(ctc_prefix_beam(np.asarray(rows), LABELS, beam_width=8, n_best=8, top_k=1))
        self.assertAlmostEqual(math.exp(scores["A"][0]), .81, places=9)
        self.assertNotIn("AA", scores)

    def test_random_lattices_match_exhaustive_enumeration(self):
        rng = np.random.default_rng(11)
        for frames in (3, 4, 5):
            rows = rng.dirichlet([3, 2, 1], size=frames)
            with self.subTest(frames=frames):
                self.assert_matches_brute_force(np.log(rows))

    def test_hand_built_double_letter_lattice(self):
        # Emissions spell A, blank, B, B, blank, E, so the only honest string is ABE.
        rows = [[-2.0, -0.1, -3.0, -3.0], [-0.1, -3.0, -3.0, -3.0], [-3.0, -3.0, -0.1, -3.0],
                [-3.0, -3.0, -0.1, -3.0], [-0.2, -3.0, -3.0, -3.0], [-2.0, -3.0, -3.0, -0.1]]
        results = ctc_prefix_beam(np.asarray(rows), LABELS + ["E"], beam_width=4, n_best=3, top_k=3)
        self.assertEqual(results[0][0], "ABE")
        self.assertEqual(results[0][2], 0.0)  # no LM, so the language score stays zero
        self.assertLess(results[0][1], 0.0)
        runner_up = ctc_prefix_beam(np.asarray(rows), LABELS + ["E"], beam_width=4, n_best=3, top_k=3)[1]
        self.assertNotEqual(runner_up[0], "ABE")

    def test_fusion_prefers_the_language_model_plausible_string(self):
        model = CharNgramLM(alphabet="AB", order=3).fit(["AB"] * 10)
        rows = np.log(np.asarray([[.02, .49, .49], [.02, .49, .49]]))
        plain = ctc_prefix_beam(rows, LABELS, beam_width=8, n_best=8, top_k=3)
        fused = ctc_prefix_beam(rows, LABELS, beam_width=8, n_best=8, top_k=3, lm=model, lm_weight=4.0)
        scores = self.scores(plain)
        self.assertAlmostEqual(scores["AB"][0], scores["BA"][0])            # acoustically tied
        self.assertIn(plain[0][0], ("A", "B"))                              # acoustics alone prefer a single letter
        self.assertLess([row[0] for row in fused].index("AB"), [row[0] for row in fused].index("BA"))
        self.assertAlmostEqual(self.scores(fused)["AB"][0], scores["AB"][0])  # fusion leaves CTC mass alone
        self.assertAlmostEqual(self.scores(fused)["AB"][1], 0.0)
        best = rescore_nbest([(text, acoustic) for text, acoustic, _ in plain], model, 4.0)[0]["text"]
        self.assertEqual(best, "AB")   # an end token is what separates AB from the shorter prefix

    def test_beam_never_invents_zero_probability_characters(self):
        half = math.log(.5)
        rows = np.asarray([[half, half, NEG], [half, half, NEG], [half, half, NEG]])
        model = CharNgramLM(alphabet="AB", order=3).fit(["BBB"] * 10)
        results = ctc_prefix_beam(rows, LABELS, beam_width=8, n_best=8, top_k=3, lm=model, lm_weight=20.0)
        self.assertTrue(results)
        self.assertTrue(all("B" not in text for text, _, _ in results))

    def test_beam_width_and_n_best_and_max_symbols_are_respected(self):
        rng = np.random.default_rng(5)
        rows = np.log(rng.dirichlet([2, 2, 2], size=6))
        narrow = ctc_prefix_beam(rows, LABELS, beam_width=2, n_best=5, top_k=3)
        self.assertLessEqual(len(narrow), 2)
        full = ctc_prefix_beam(rows, LABELS, beam_width=8, n_best=3, top_k=3)
        self.assertEqual(len(full), 3)
        capped = ctc_prefix_beam(rows, LABELS, beam_width=8, n_best=8, top_k=3, max_symbols=2)
        self.assertTrue(all(len(text) <= 2 for text, _, _ in capped))
        with self.assertRaises(ValueError):
            ctc_prefix_beam(rows, LABELS[:2], beam_width=4)


if __name__ == "__main__":
    unittest.main()
