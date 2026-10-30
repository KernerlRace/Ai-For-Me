"""Retraining the intent classifier on labelled commands, behind a gate.

    python -m aiforme retrain                     # the assistant's embedder, e5
    python -m aiforme retrain --embedder tfidf

The training set is the hand-written commands plus every label: a label replaces a hand-written example with the
same text, and a command whose text is also a test phrase is left out, so the test set stays held out. A candidate
trained on the pending labels too is compared with the current model in two ways:

- the test set (aiforme/voice_orchestrator/data/test.jsonl): it may lose at most one phrase;
- the pending labels themselves, the commands I actually said: the current model never trained on them, so its
  accuracy there is honest; the candidate is scored on them out of fold — trained on all but a fold, scored on it —
  and must be at least as good.

The out-of-scope threshold is chosen once, by cross-validation on the hand-written commands, and carried from
version to version. Chosen again on every retraining it moves by noise — six labels moved it from 0.20 to 0.22 —
and the gate then punishes the move: at 0.22 the candidate lost six test phrases, at 0.20 none.

Accepted, the candidate becomes the next version and the pending labels part of it; the assistant reloads on
`intent.model_updated`. Rejected, it is recorded with its numbers and the labels stay pending — a wrong label can
then be found and undone in the dashboard.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from aiforme.active_learning.store import Feedback, Label, Version
from aiforme.core import Event, EventBus
from aiforme.voice_orchestrator.evaluate import score
from aiforme.voice_orchestrator.nlu import (
    TEST,
    TRAIN,
    Embedder,
    Example,
    IntentClassifier,
    load_examples,
    make_embedder,
    normalize,
)

MODEL_UPDATED = "intent.model_updated"


class NothingToLearn(Exception):
    """No pending labels."""


def training_set(labels: list[Label], base: list[Example], test: list[Example]) -> list[Example]:
    held_out = {normalize(e.text) for e in test}
    merged = {normalize(e.text): e for e in base}
    for lab in labels:
        key = normalize(lab.text)
        if key not in held_out:
            merged[key] = Example(lab.text, lab.intent, lab.slots)
    return list(merged.values())


class _Memo:
    """A frozen encoder that remembers what it has encoded: retraining encodes the same few hundred commands for
    every fit and every fold, and a sentence encoder is the slow part."""
    trainable = False

    def __init__(self, inner: Embedder):
        self.inner, self.name, self._cache = inner, inner.name, {}

    def fit(self, texts: list[str], labels: np.ndarray) -> None:
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            self._cache.update(zip(missing, self.inner.encode(missing), strict=True))
        return np.stack([self._cache[t] for t in texts])


def embedders(name: str, device: str | None = None):
    """A factory of embedders: a fresh one per fit for those trained on the data, one shared one for the frozen."""
    first = make_embedder(name, device)
    if first.trainable:
        return lambda: make_embedder(name, device)
    shared = _Memo(first)
    return lambda: shared


def current_model(feedback: Feedback, embedder: str = "e5", device: str | None = None
                  ) -> tuple[IntentClassifier, int]:
    """The classifier of the current version — the hand-written commands and the accepted labels — and its id."""
    data = training_set(feedback.labels(pending=False), load_examples(TRAIN), load_examples(TEST))
    clf = IntentClassifier(make_embedder(embedder, device), threshold=current_threshold(feedback)).fit(data)
    return clf, feedback.current_version()


def current_threshold(feedback: Feedback) -> float | None:
    """The threshold of the current version; None before the first retraining, when it is still to be chosen."""
    current = feedback.current_version()
    return next((v.metrics.get("threshold") for v in feedback.versions() if v.id == current), None)


def retrain(feedback: Feedback, embedder: str = "e5", device: str | None = None, folds: int = 5) -> Version:
    base, test = load_examples(TRAIN), load_examples(TEST)
    included, pending = feedback.labels(pending=False), feedback.labels(pending=True)
    if not pending:
        raise NothingToLearn("no pending labels")
    new = embedders(embedder, device)
    current = IntentClassifier(new(), threshold=current_threshold(feedback)).fit(training_set(included, base, test))
    data = training_set(included + pending, base, test)
    candidate = IntentClassifier(new(), threshold=current.threshold).fit(data)

    texts = [e.text for e in test]
    test_before, test_after = score(test, current.predict(texts)), score(test, candidate.predict(texts))
    said = [Example(lab.text, lab.intent, lab.slots) for lab in pending]
    real_before = score(said, current.predict([e.text for e in said]))
    real_after = score(said, _out_of_fold(pending, included, base, test, new, candidate.threshold, folds))

    if not feedback.versions():                 # version 0 gets its test numbers from the first retraining
        feedback.add_version(embedder, True, [], len(training_set([], base, test)),
                             {"test": test_before, "real": None, "threshold": current.threshold}, version_id=0)
    lost = round((test_before["accuracy"] - test_after["accuracy"]) * len(test))
    checks = {"test_phrases_lost": lost, "test_ok": lost <= 1,
              "real_ok": real_after["accuracy"] >= real_before["accuracy"]}
    accepted = checks["test_ok"] and checks["real_ok"]
    metrics = {"test": test_after, "test_before": test_before, "real": real_after, "real_before": real_before,
               "real_labels": len(pending), "threshold": candidate.threshold, **checks}
    return feedback.add_version(embedder, accepted, pending, len(data), metrics)


def _out_of_fold(pending, included, base, test, new, threshold, folds):
    """Each pending label predicted by a model trained on everything but its fold, at the candidate's threshold."""
    parses = [None] * len(pending)
    k = min(folds, len(pending))
    splits = KFold(k, shuffle=True, random_state=0).split(pending) if k > 1 else [([], [0])]
    for fit_idx, val_idx in splits:
        rest = [pending[i] for i in fit_idx]
        clf = IntentClassifier(new(), threshold=threshold).fit(training_set(included + rest, base, test))
        for i, p in zip(val_idx, clf.predict([pending[i].text for i in val_idx]), strict=True):
            parses[i] = p
    return parses


def retrain_and_announce(feedback: Feedback, bus: EventBus | None, embedder: str = "e5",
                         device: str | None = None) -> Version:
    version = retrain(feedback, embedder, device)
    if version.accepted and bus is not None:
        bus.publish(Event(MODEL_UPDATED, {"version": version.id}, source="active_learning"))
    return version

