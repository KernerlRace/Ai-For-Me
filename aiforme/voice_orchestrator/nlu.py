"""Intent recognition, Phase 1 baseline: a frozen text embedding + logistic regression, no fine-tuning.

Four embeddings are compared (see evaluate.py). Two are built from the training commands alone: character
n-gram TF-IDF, which runs in CI, and a character CNN trained from scratch. Two are pretrained multilingual
sentence encoders, for reference. The classifier is retrained at start-up — a few hundred
examples take seconds — so there is no model file to keep in sync with the data.

Out-of-scope requests ("order a pizza") are an explicit class with their own examples, and on top of that a
confidence threshold: an in-scope intent whose probability is below it becomes out-of-scope too. The threshold
is chosen on out-of-fold predictions on the training set, never on test.

Slots come from carrier phrases learned from the training data: in "open my steam" with the slot "steam" the
carrier prefix is "open my". At inference the longest known prefix (and suffix) of the predicted intent is cut
off; when none matches, leading words that only ever appear in that intent's carriers are dropped instead.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from aiforme.voice_orchestrator.intents import INTENTS, OUT_OF_SCOPE

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN = DATA_DIR / "train.jsonl"
TEST = DATA_DIR / "test.jsonl"

# words dropped from the edges of an app name or a search query, never from dictated text
FILLERS = frozenset(["пожалуйста", "please", "мне", "мой", "моя", "мою", "мое", "ка", "a", "an", "the", "my"])
POLITE = frozenset(["пожалуйста", "please"])


def normalize(text: str) -> str:
    """Lowercase, ё → е, punctuation out (apostrophes inside words stay: "i'll"), single spaces.
    Whisper writes "Открой Telegram." where the training data says "открой telegram"."""
    text = text.lower().replace("ё", "е").replace("’", "'")
    text = re.sub(r"[-‐–—_/]", " ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    return " ".join(text.split())


def language(text: str) -> str:
    return "ru" if re.search(r"[а-яё]", text, re.IGNORECASE) else "en"


@dataclass(frozen=True)
class Example:
    text: str
    intent: str
    slots: dict[str, str] = field(default_factory=dict)


def load_examples(path: str | Path) -> list[Example]:
    examples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            examples.append(Example(row["text"], row["intent"], row.get("slots", {})))
    return examples


@dataclass(frozen=True)
class Parse:
    text: str
    intent: str
    confidence: float           # probability of `intent`; for out-of-scope, of the best in-scope guess
    slots: dict[str, str]
    best_guess: str = ""        # the in-scope intent that lost to out-of-scope, if any


# --- embeddings ---------------------------------------------------------------------------------------

class Embedder(Protocol):
    name: str
    trainable: bool             # fitted on the training data (then refitted inside every CV fold)

    def fit(self, texts: list[str], labels: np.ndarray) -> None: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...


class TfidfEmbedder:
    """Character n-grams inside word boundaries: robust to inflection (дискорд / дискорда) and typos."""
    name = "tfidf"
    trainable = True

    def __init__(self) -> None:
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)

    def fit(self, texts: list[str], labels: np.ndarray) -> None:
        self._vec.fit(texts)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._vec.transform(texts)


class SentenceEmbedder:
    """A frozen multilingual sentence encoder from Hugging Face (sentence-transformers, the `nlu` extra)."""
    trainable = False

    def __init__(self, name: str, model: str, prefix: str = "", device: str | None = None):
        self.name, self.model_name, self.prefix, self.device = name, model, prefix, device
        self._model = None

    def fit(self, texts: list[str], labels: np.ndarray) -> None:
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            try:    # once downloaded, start without asking the Hub for updates
                self._model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
            except OSError:
                self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model.encode([self.prefix + t for t in texts], normalize_embeddings=True,
                                  batch_size=64, show_progress_bar=False)


class CharCNNEmbedder:
    """A character CNN trained from scratch on the training commands, no pretrained weights.

    Characters are embedded, convolved with filters 2–5 characters wide and max-pooled over the text; the
    network is trained to predict the intent, and its 256 pooled features become the embedding the logistic
    regression sees, like the other embedders'. Typos are simulated in training: each character is dropped or
    swapped with its neighbour with probability 0.03. Needs PyTorch (the `train` extra).
    """
    name = "charcnn"
    trainable = True

    def __init__(self, epochs: int = 80, seed: int = 0):
        self.epochs, self.seed = epochs, seed

    def fit(self, texts: list[str], labels: np.ndarray) -> None:
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._chars = {c: i + 2 for i, c in enumerate(sorted({c for t in texts for c in t}))}  # 0 pad, 1 unknown
        classes = sorted(set(labels))
        y = torch.tensor([classes.index(lab) for lab in labels])
        self._net = _CharCNN(len(self._chars) + 2, len(classes))
        opt = torch.optim.AdamW(self._net.parameters(), lr=3e-3, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            self._net.train()
            for b in torch.randperm(len(texts)).split(32):
                batch = self._ids([_typos(texts[i], rng) for i in b.tolist()])
                loss = loss_fn(self._net(batch), y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch
        self._net.eval()
        with torch.no_grad():
            feats = self._net.features(self._ids(texts)).numpy()
        return feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)

    def _ids(self, texts: list[str]):
        import torch
        width = max(8, max(len(t) for t in texts))
        ids = np.zeros((len(texts), width), dtype=np.int64)
        for row, t in enumerate(texts):
            ids[row, :len(t)] = [self._chars.get(c, 1) for c in t]
        return torch.tensor(ids)


def _typos(text: str, rng: np.random.Generator, p: float = 0.03) -> str:
    chars = list(text)
    i = 0
    while i < len(chars):
        r = rng.random()
        if r < p and len(chars) > 3:
            del chars[i]
            continue
        if r < 2 * p and i + 1 < len(chars):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        i += 1
    return "".join(chars)


def _CharCNN(vocab: int, classes: int, dim: int = 32, filters: int = 64, widths=(2, 3, 4, 5)):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, dim, padding_idx=0)
            self.convs = nn.ModuleList(nn.Conv1d(dim, filters, w, padding=w // 2) for w in widths)
            self.drop = nn.Dropout(0.3)
            self.out = nn.Linear(filters * len(widths), classes)

        def features(self, ids):
            x = self.embed(ids).transpose(1, 2)
            mask = (ids == 0)[:, None, :]
            pooled = []
            for conv in self.convs:
                h = torch.relu(conv(x))[:, :, : ids.shape[1]]
                pooled.append(h.masked_fill(mask, -1e4).max(dim=2).values)
            return torch.cat(pooled, dim=1)

        def forward(self, ids):
            return self.out(self.drop(self.features(ids)))

    return Net()


def make_embedder(name: str, device: str | None = None) -> Embedder:
    if name == "tfidf":
        return TfidfEmbedder()
    if name == "charcnn":
        return CharCNNEmbedder()
    if name == "minilm":
        return SentenceEmbedder(name, "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                device=device)
    if name == "e5":     # E5 models expect a "query: " prefix on every input
        return SentenceEmbedder(name, "intfloat/multilingual-e5-small", prefix="query: ", device=device)
    raise ValueError(f"unknown embedder {name!r}: expected one of {EMBEDDERS}")


EMBEDDERS = ("tfidf", "charcnn", "minilm", "e5")


# --- slots --------------------------------------------------------------------------------------------

class SlotExtractor:
    def fit(self, examples: list[Example]) -> SlotExtractor:
        self.prefixes: dict[str, set[str]] = defaultdict(set)
        self.suffixes: dict[str, set[str]] = defaultdict(set)
        self.vocab: dict[str, set[str]] = defaultdict(set)
        for ex in examples:
            slot = INTENTS.get(ex.intent)
            if not slot or slot not in ex.slots:
                continue
            text, value = normalize(ex.text), normalize(ex.slots[slot])
            start = f" {text} ".find(f" {value} ")
            if start < 0:
                raise ValueError(f"slot {value!r} is not a phrase of {text!r}")
            prefix, suffix = text[:start].strip(), text[start + len(value):].strip()
            self.prefixes[ex.intent].add(prefix)
            self.suffixes[ex.intent].add(suffix)
            self.vocab[ex.intent].update(prefix.split())
        return self

    def extract(self, intent: str, text: str) -> dict[str, str]:
        slot = INTENTS.get(intent)
        if not slot:
            return {}
        words = normalize(text).split()
        prefix = _longest_match(words, self.prefixes[intent], at_start=True)
        if prefix:
            words = words[prefix:]
        else:
            while words and words[0] in self.vocab[intent]:
                words = words[1:]
        suffix = _longest_match(words, self.suffixes[intent], at_start=False)
        if suffix:
            words = words[:-suffix]
        edge = POLITE if slot == "text" else FILLERS
        while words and words[0] in edge:
            words = words[1:]
        while slot != "text" and words and words[-1] in edge:
            words = words[:-1]
        return {slot: " ".join(words)} if words else {}


def _longest_match(words: list[str], phrases: set[str], at_start: bool) -> int:
    """Length in words of the longest phrase that `words` starts (or ends) with, leaving something after it."""
    best = 0
    for phrase in phrases:
        n = len(phrase.split())
        if 0 < n < len(words) and n > best:
            part = words[:n] if at_start else words[-n:]
            if " ".join(part) == phrase:
                best = n
    return best


# --- classifier ---------------------------------------------------------------------------------------

class IntentClassifier:
    def __init__(self, embedder: Embedder, C: float = 10.0, threshold: float | None = None):
        self.embedder, self.C, self.threshold = embedder, C, threshold
        self.slots = SlotExtractor()

    def fit(self, examples: list[Example], folds: int = 5, seed: int = 0) -> IntentClassifier:
        texts = [normalize(e.text) for e in examples]
        labels = np.array([e.intent for e in examples])
        if self.threshold is None:
            self.threshold, self.cv_report = self._fit_threshold(texts, labels, folds, seed)
        self.embedder.fit(texts, labels)
        self._clf = self._new_clf().fit(self.embedder.encode(texts), labels)
        self.slots.fit(examples)
        return self

    def predict(self, texts: list[str]) -> list[Parse]:
        norm = [normalize(t) for t in texts]
        probs = self._clf.predict_proba(self.embedder.encode(norm))
        return [self._decide(text, p) for text, p in zip(texts, probs, strict=True)]

    def predict_one(self, text: str) -> Parse:
        return self.predict([text])[0]

    def _decide(self, text: str, p: np.ndarray) -> Parse:
        classes = self._clf.classes_
        intent, conf = _apply_threshold(classes, p, self.threshold)
        if intent == OUT_OF_SCOPE:
            in_scope = classes != OUT_OF_SCOPE
            best = int(np.argmax(np.where(in_scope, p, -1)))
            return Parse(text, OUT_OF_SCOPE, float(p[best]), {}, best_guess=str(classes[best]))
        return Parse(text, intent, conf, self.slots.extract(intent, text))

    def _new_clf(self) -> LogisticRegression:
        return LogisticRegression(C=self.C, max_iter=5000, class_weight="balanced")

    def _fit_threshold(self, texts, labels, folds, seed) -> tuple[float, dict]:
        """Out-of-fold probabilities on the training set; the threshold maximises the mean of in-scope
        accuracy and out-of-scope recall. Ties go to the lower threshold, which rejects less."""
        oof = None
        X_all = None if self.embedder.trainable else self.embedder.encode(texts)
        for train_idx, val_idx in StratifiedKFold(folds, shuffle=True, random_state=seed).split(texts, labels):
            if self.embedder.trainable:
                self.embedder.fit([texts[i] for i in train_idx], labels[train_idx])
                X_tr, X_val = (self.embedder.encode([texts[i] for i in idx]) for idx in (train_idx, val_idx))
            else:
                X_tr, X_val = X_all[train_idx], X_all[val_idx]
            clf = self._new_clf().fit(X_tr, labels[train_idx])
            if oof is None:
                classes = clf.classes_
                oof = np.zeros((len(texts), len(classes)))
            oof[val_idx] = clf.predict_proba(X_val)
        in_scope = labels != OUT_OF_SCOPE
        best = (-1.0, 0.0, {})
        for t in np.round(np.arange(0.0, 0.96, 0.01), 2):
            pred = np.array([_apply_threshold(classes, p, t)[0] for p in oof])
            acc_in = float(np.mean(pred[in_scope] == labels[in_scope]))
            rec_oos = float(np.mean(pred[~in_scope] == OUT_OF_SCOPE))
            score = (acc_in + rec_oos) / 2
            if score > best[0] + 1e-9:
                best = (score, float(t), {"in_scope_accuracy": acc_in, "oos_recall": rec_oos, "score": score})
        return best[1], best[2]


def _apply_threshold(classes: np.ndarray, p: np.ndarray, threshold: float) -> tuple[str, float]:
    i = int(np.argmax(p))
    intent, conf = str(classes[i]), float(p[i])
    if intent != OUT_OF_SCOPE and conf < threshold:
        return OUT_OF_SCOPE, conf
    return intent, conf


def train(embedder: str = "tfidf", device: str | None = None) -> IntentClassifier:
    return IntentClassifier(make_embedder(embedder, device)).fit(load_examples(TRAIN))
