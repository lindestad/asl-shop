"""Additive recognition diagnostics: alignment error types and target-length accuracy.

The keys a report already carries (clips, characters, errors, exact, cer,
exact_sequence_accuracy) are kept identical so existing readers keep working; the
error-type and length-bin numbers are extra keys next to them.
"""
from collections import Counter

LENGTH_BINS = ((1, 4), (5, None))  # the paper's short/long target split.


def bin_label(low, high):
    return f"{low}-{high}" if high else f"{low}+"


def align_counts(expected, actual):
    """Count deletions, substitutions and insertions in one minimum-cost alignment.

    Deletions drop a target character, insertions add a character the target does
    not have; they are read straight off the alignment rather than estimated from
    the edit distance. Equal-cost ties prefer substitution, then deletion, so a
    conservative decoder is charged deletions instead of delete-plus-insert pairs.
    D + S + I always equals the edit distance used for CER.
    """
    rows = len(expected) + 1
    columns = len(actual) + 1
    cost = [[0] * columns for _ in range(rows)]
    moves = [[""] * columns for _ in range(rows)]
    for i in range(1, rows):
        cost[i][0], moves[i][0] = i, "D"
    for j in range(1, columns):
        cost[0][j], moves[0][j] = j, "I"
    for i in range(1, rows):
        left, previous, expected_character = cost[i], cost[i - 1], expected[i - 1]
        move_row, previous_move = moves[i], moves[i - 1]
        for j in range(1, columns):
            substitution = previous[j - 1] + (expected_character != actual[j - 1])
            deletion = previous[j] + 1
            insertion = left[j - 1] + 1
            if deletion < substitution:
                substitution, move = deletion, "D"
            else:
                move = "S"
            if insertion < substitution:
                substitution, move = insertion, "I"
            left[j], move_row[j] = substitution, move
    deletions = substitutions = insertions = 0
    i, j = len(expected), len(actual)
    while i or j:
        move = moves[i][j]
        if move == "S":
            substitutions += expected[i - 1] != actual[j - 1]
            i, j = i - 1, j - 1
        elif move == "D":
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return deletions, substitutions, insertions


class ErrorTally:
    """Running per-group totals that fold in the error-type and length breakdown."""

    def __init__(self):
        self.clips = 0
        self.characters = 0
        self.errors = 0
        self.exact = 0
        self.deletions = 0
        self.substitutions = 0
        self.insertions = 0
        self.lengths = Counter()
        self.length_errors = Counter()
        self.length_exact = Counter()

    def add(self, expected, actual):
        deletions, substitutions, insertions = align_counts(expected, actual)
        errors = deletions + substitutions + insertions
        self.clips += 1
        self.characters += len(expected)
        self.errors += errors
        self.exact += expected == actual
        self.deletions += deletions
        self.substitutions += substitutions
        self.insertions += insertions
        self.lengths[len(expected)] += 1
        self.length_errors[len(expected)] += errors
        self.length_exact[len(expected)] += expected == actual
        return errors

    def merge(self, other):
        """Fold another tally's totals into this one (both kept for shared-pair caching)."""
        self.clips += other.clips
        self.characters += other.characters
        self.errors += other.errors
        self.exact += other.exact
        self.deletions += other.deletions
        self.substitutions += other.substitutions
        self.insertions += other.insertions
        self.lengths.update(other.lengths)
        self.length_errors.update(other.length_errors)
        self.length_exact.update(other.length_exact)
        return self

    def summary(self, length_bins=LENGTH_BINS):
        cer = self.errors / self.characters if self.characters else None
        result = dict(clips=self.clips, characters=self.characters, errors=self.errors, exact=self.exact,
                      cer=cer, letter_accuracy=None if cer is None else 1 - cer,
                      exact_sequence_accuracy=self.exact / self.clips if self.clips else None,
                      deletions=self.deletions, substitutions=self.substitutions, insertions=self.insertions)
        if self.characters:
            result.update(deletion_rate=self.deletions / self.characters,
                          substitution_rate=self.substitutions / self.characters,
                          insertion_rate=self.insertions / self.characters)
        result["length_clips"] = {bin_label(low, high): sum(count for length, count in self.lengths.items()
                                                            if length >= low and not (high and length > high))
                                  for low, high in length_bins}
        result["length_exact"] = {bin_label(low, high): sum(count for length, count in self.length_exact.items()
                                                            if length >= low and not (high and length > high))
                                  for low, high in length_bins}
        result["length_accuracy"] = {label: result["length_exact"][label] / result["length_clips"][label]
                                     if result["length_clips"][label] else None
                                     for label in result["length_clips"]}
        result["length_cer"] = {bin_label(low, high): (sum(count for length, count in self.length_errors.items()
                                                          if length >= low and not (high and length > high))
                                                       / (sum(length * count for length, count in self.lengths.items()
                                                             if length >= low and not (high and length > high)) or 1))
                                for low, high in length_bins}
        return result


def tally(pairs, length_bins=LENGTH_BINS, key=None):
    """Summarize an iterable of (expected, actual) pairs into one report block."""
    group = ErrorTally()
    for expected, actual in pairs:
        group.add(expected, actual)
    summary = group.summary(length_bins)
    if key is not None:
        summary["group"] = key
    return summary
