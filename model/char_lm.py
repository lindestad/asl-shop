"""Character n-gram language model with stupid backoff, plus rescoring helpers.

Only training-split text may be fit here; transcripts_from_files drops any clip whose
name is in the excluded set so a dev or test split cannot enter the counts.
"""
import gzip
import json
import math
from collections import Counter
from pathlib import Path

START = "\u0002"  # sentence context that is never predicted
END = "\u0003"    # end-of-transcript token, predicted like a character
UNKNOWN = "\u0004"
NEG = float("-inf")
TEXT_FIELDS = ("text", "phrase", "transcript")
NAME_FIELDS = ("name", "clipFilename")
CACHE_LIMIT = 400_000


def log_add(left, right):
    if left == NEG:
        return right
    if right == NEG:
        return left
    high = left if left > right else right
    return high + math.log1p(math.exp(-abs(left - right)))


def transcripts_from_files(paths, *, exclude_names=()):
    """Read transcripts from packed clip JSON, FSboard metadata JSON or plain text files.

    Records whose clip name also appears in exclude_names are dropped, which keeps
    held-out clips out of the model even when a source file overlaps several splits.
    """
    exclude = set(exclude_names)
    texts, sources, excluded = [], [], 0
    for path in map(Path, paths):
        if path.suffix == ".txt":
            records = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            records = json.loads(path.read_text(encoding="utf-8"))
        kept, characters = 0, 0
        for record in records:
            if isinstance(record, str):
                text, name = record, None
            else:
                text = next((record[field] for field in TEXT_FIELDS if record.get(field) is not None), None)
                name = next((record[field] for field in NAME_FIELDS if record.get(field) is not None), None)
                if name and name.endswith(".mp4"):
                    name = name[:-4]
            if text is None:
                continue
            if name is not None and name in exclude:
                excluded += 1
                continue
            text = text.strip().upper()
            if not text:
                continue
            texts.append(text)
            characters += len(text)
            kept += 1
        sources.append(dict(path=str(path), records=kept, characters=characters))
    return dict(texts=texts, sources=sources, excluded_records=excluded)


class CharNgramLM:
    """Stupid-backoff n-gram over characters, with smoothed unigrams as the floor."""

    def __init__(self, order=5, backoff_weight=.4, alphabet=(), unigram_pseudo=.1):
        if not 1 <= order <= 12:
            raise ValueError("LM order must be between 1 and 12.")
        if not 0 < backoff_weight < 1:
            raise ValueError("Backoff weight must be between 0 and 1.")
        self.order = order
        self.backoff_weight = backoff_weight
        self.unigram_pseudo = unigram_pseudo
        self.alphabet = "".join(dict.fromkeys(alphabet))
        self.counts = {}       # context string -> {token: count}
        self.totals = Counter()  # context string -> next-token total
        self.unigrams = Counter()
        self.trained_tokens = 0
        self.lexicon = ()
        self._log_cache = {}

    def normalize(self, character):
        character = character.upper()
        if self.alphabet and character not in self.alphabet:
            return UNKNOWN
        return character

    def fit(self, texts):
        """Add transcript counts; safe to call more than once on training text only."""
        for text in texts:
            tokens = [START] + [self.normalize(character) for character in text] + [END]
            self.unigrams.update(tokens[1:])
            self.trained_tokens += len(tokens) - 1
            for index in range(1, len(tokens)):
                context = "".join(tokens[max(0, index - self.order + 1):index])
                row = self.counts.get(context)
                if row is None:
                    row = self.counts[context] = {}
                row[tokens[index]] = row.get(tokens[index], 0) + 1
                self.totals[context] += 1
        seen = set(self.unigrams) | {END, UNKNOWN}
        self.lexicon = "".join(sorted(seen | set(self.alphabet) - {START}))
        self._log_cache = {}
        return self

    def unigram(self, token):
        floor = self.unigram_pseudo * max(len(self.lexicon), 1)
        total = self.trained_tokens + floor
        return (self.unigrams.get(token, 0) + self.unigram_pseudo) / total

    def probability(self, context, token):
        """P(token | context) with multiplicative stupid backoff at every shorter order."""
        weight = 1.0
        while context:
            row = self.counts.get(context)
            count = row.get(token, 0) if row else 0
            if count:
                return weight * count / self.totals[context]
            weight *= self.backoff_weight
            context = context[1:]
        return weight * self.unigram(token)

    def log_prob(self, context, token):
        key = (context, token)
        value = self._log_cache.get(key)
        if value is None:
            probability = self.probability(context, token)
            value = math.log(probability) if probability > 0 else NEG
            if len(self._log_cache) >= CACHE_LIMIT:
                self._log_cache = {}
            self._log_cache[key] = value
        return value

    def context_for(self, prefix):
        """LM context for predicting the character after prefix, keeping the START token."""
        return (START + prefix)[-self.order + 1:] if self.order > 1 else ""

    def extend_log_prob(self, prefix, character):
        """Extra log probability earned by appending character to prefix.

        prefix must already hold normalized characters, as produced by score and by
        the CTC beam's model labels; character is normalized here.
        """
        return self.log_prob(self.context_for(prefix), self.normalize(character))

    def score(self, text, *, include_end=True):
        """Total log probability the model assigns to text, in nats."""
        total, prefix = 0.0, ""
        for character in text:
            character = self.normalize(character)
            total += self.extend_log_prob(prefix, character)
            prefix += character
        if include_end:
            total += self.log_prob(self.context_for(prefix), END)
        return total

    def token_count(self, text, *, include_end=True):
        return len(text) + int(include_end)

    def perplexity(self, texts, *, include_end=True):
        """exp(mean nats of negative log probability) over scored transcript tokens."""
        total = tokens = 0
        for text in texts:
            total += self.score(text, include_end=include_end)
            tokens += self.token_count(text, include_end=include_end)
        return math.exp(-total / tokens) if tokens else None

    def ngram_count(self, text):
        """Raw count of the exact string as a trained n-gram, for leakage checks."""
        return self.counts.get(text[:-1], {}).get(text[-1], 0)

    def to_dict(self):
        return dict(order=self.order, backoff_weight=self.backoff_weight, unigram_pseudo=self.unigram_pseudo,
                    alphabet=self.alphabet, trained_tokens=self.trained_tokens,
                    unigrams=dict(self.unigrams), counts={context: dict(row) for context, row in self.counts.items()})

    @classmethod
    def from_dict(cls, payload):
        model = cls(order=payload["order"], backoff_weight=payload["backoff_weight"],
                    alphabet=payload.get("alphabet", ""), unigram_pseudo=payload.get("unigram_pseudo", .1))
        model.counts = {context: dict(row) for context, row in payload["counts"].items()}
        model.totals = Counter({context: sum(row.values()) for context, row in model.counts.items()})
        model.unigrams = Counter(payload["unigrams"])
        model.trained_tokens = payload["trained_tokens"]
        model.lexicon = "".join(sorted(set(model.unigrams) | {END, UNKNOWN} | set(model.alphabet) - {START}))
        return model

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as file:
            json.dump(self.to_dict(), file)

    @classmethod
    def load(cls, path):
        with gzip.open(path, "rt", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))


def rescore_nbest(hypotheses, lm, lm_weight, *, include_end=True):
    """Rank (text, acoustic log probability) hypotheses by acoustic + lm_weight * LM."""
    ranked = []
    for text, acoustic in hypotheses:
        language = lm.score(text, include_end=include_end)
        ranked.append(dict(text=text, acoustic=acoustic, language=language, total=acoustic + lm_weight * language))
    ranked.sort(key=lambda row: -row["total"])
    return ranked


def best_rescored(hypotheses, lm, lm_weight, *, include_end=True):
    return rescore_nbest(hypotheses, lm, lm_weight, include_end=include_end)[0]["text"]


def ctc_prefix_beam(log_probs, labels, *, blank=0, beam_width=8, n_best=1, lm=None, lm_weight=0.0,
                    top_k=8, max_symbols=None):
    """CTC prefix beam search with optional character-LM shallow fusion.

    log_probs is a T x V array of log probabilities and labels maps a column index to a
    character. Returns (text, ctc log probability, LM log probability without the end
    token) ordered by the score actually used for decoding, so the same n-best list can
    be re-ranked afterwards at another LM weight.
    """
    import numpy as np

    frames, symbols = log_probs.shape
    if len(labels) != symbols:
        raise ValueError("CTC labels must cover every output column.")
    beam_width = max(1, beam_width)
    keep = min(max(1, top_k), symbols - 1)
    columns = (np.argsort(-log_probs, axis=1)[:, :keep + 1] if keep + 1 < symbols
               else np.tile(np.arange(symbols), (frames, 1)))
    label_index = {label: index for index, label in enumerate(labels) if index != blank}
    extend_score = lm.extend_log_prob if lm else None
    state = {"": [0.0, NEG, 0.0]}  # prefix -> [blank-ending, symbol-ending, LM score]
    for frame in range(frames):
        row = log_probs[frame]
        emissions = row.tolist() if hasattr(row, "tolist") else list(row)
        blank_log = emissions[blank]
        candidates = [index for index in columns[frame].tolist() if index != blank]
        following = {}
        for prefix, (probability_blank, probability_nonblank, language) in state.items():
            total = log_add(probability_blank, probability_nonblank)
            if total == NEG:
                continue
            last = prefix[-1] if prefix else ""
            entry = following.get(prefix)
            if entry is None:
                following[prefix] = entry = [NEG, NEG, language]
            entry[0] = log_add(entry[0], total + blank_log)
            repeat = label_index.get(last, blank)
            if last and repeat != blank:
                # Repeating a symbol inside one emission run collapses back to this prefix.
                entry[1] = log_add(entry[1], probability_nonblank + emissions[repeat])
            for index in candidates:
                character = labels[index]
                # Only an intervening blank turns a repeated symbol into a real doubled one.
                value = probability_blank + emissions[index] if character == last else total + emissions[index]
                if value == NEG or (max_symbols is not None and len(prefix) >= max_symbols):
                    continue
                extend = prefix + character
                target = following.get(extend)
                if target is None:
                    extra = extend_score(prefix, character) if extend_score else 0.0
                    following[extend] = target = [NEG, NEG, language + extra]
                target[1] = log_add(target[1], value)
        if len(following) > beam_width:
            following = dict(sorted(following.items(),
                                    key=lambda row: -(log_add(row[1][0], row[1][1]) + lm_weight * row[1][2]))[:beam_width])
        state = following
    results = [(prefix, log_add(entry[0], entry[1]), entry[2]) for prefix, entry in state.items()]
    results = [row for row in results if row[1] != NEG]
    results.sort(key=lambda row: -(row[1] + lm_weight * row[2]))
    return results[:max(1, n_best)]
