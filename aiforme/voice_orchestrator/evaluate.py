"""Scores the intent baselines on the held-out test set.

    python -m aiforme.voice_orchestrator.evaluate                 # all three embeddings
    python -m aiforme.voice_orchestrator.evaluate --embedder tfidf

Writes eval/results/intent_<embedder>.json (metrics and every error) and prints one comparison table.
The test phrases were written separately from the training ones, with other wording, Whisper-style
capitals and punctuation, slang and apps the training data never names; but both were written by the
same person during development, so they are an optimistic estimate. Real transcribed speech comes next.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from aiforme.voice_orchestrator.intents import INTENTS, OUT_OF_SCOPE
from aiforme.voice_orchestrator.nlu import (
    EMBEDDERS,
    TEST,
    TRAIN,
    IntentClassifier,
    language,
    load_examples,
    make_embedder,
    normalize,
)

RESULTS = Path(__file__).resolve().parents[2] / "eval" / "results"


def evaluate(embedder: str, device: str | None = None) -> dict:
    train, test = load_examples(TRAIN), load_examples(TEST)
    t0 = time.perf_counter()
    clf = IntentClassifier(make_embedder(embedder, device)).fit(train)
    fit_s = time.perf_counter() - t0
    clf.predict([e.text for e in test[:4]])            # warm-up, so latency excludes model loading
    t0 = time.perf_counter()
    parses = [clf.predict_one(e.text) for e in test]  # one at a time, as the assistant will call it
    latency_ms = (time.perf_counter() - t0) / len(test) * 1000
    report = {
        "embedder": embedder,
        "train_examples": len(train),
        "test_examples": len(test),
        "threshold": clf.threshold,
        "cv_on_train": clf.cv_report,
        **score(test, parses),
        "fit_seconds": round(fit_s, 2),
        "latency_ms": round(latency_ms, 2),
        "errors": [
            {"text": e.text, "gold": e.intent, "pred": p.intent, "confidence": round(p.confidence, 3),
             "gold_slots": e.slots, "pred_slots": p.slots, "best_guess": p.best_guess}
            for e, p in zip(test, parses, strict=True)
            if p.intent != e.intent or (INTENTS.get(e.intent) and _slot(p, e)
                                        != normalize(e.slots[INTENTS[e.intent]]))
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"intent_{embedder}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def score(test: list, parses: list) -> dict:
    """Intent and slot metrics of `parses` against the labelled `test` examples, in the same order."""
    gold = np.array([e.intent for e in test])
    pred = np.array([p.intent for p in parses])
    langs = np.array([language(e.text) for e in test])
    in_scope = gold != OUT_OF_SCOPE

    def acc(mask):
        return float(np.mean(pred[mask] == gold[mask])) if mask.any() else None

    slot_rows = [(e, p) for e, p in zip(test, parses, strict=True) if INTENTS.get(e.intent)]
    slot_ok = [p.intent == e.intent and _slot(p, e) == normalize(e.slots[INTENTS[e.intent]])
               for e, p in slot_rows]
    slot_given_intent = [ok for ok, (e, p) in zip(slot_ok, slot_rows, strict=True) if p.intent == e.intent]

    return {
        "accuracy": acc(np.ones_like(in_scope)),
        "in_scope_accuracy": acc(in_scope),
        "in_scope_accuracy_ru": acc(in_scope & (langs == "ru")),
        "in_scope_accuracy_en": acc(in_scope & (langs == "en")),
        "oos_recall": acc(~in_scope),
        "false_rejections": float(np.mean(pred[in_scope] == OUT_OF_SCOPE)) if in_scope.any() else None,
        "slot_end_to_end": float(np.mean(slot_ok)) if slot_ok else None,
        "slot_given_intent": float(np.mean(slot_given_intent)) if slot_given_intent else None,
    }


def _slot(parse, example) -> str:
    return parse.slots.get(INTENTS[example.intent], "")


def table(reports: list[dict]) -> str:
    rows = [
        ("Intent accuracy, all test phrases", "accuracy"),
        ("In-scope intent accuracy", "in_scope_accuracy"),
        ("— Russian / English", ("in_scope_accuracy_ru", "in_scope_accuracy_en")),
        ("Out-of-scope recall", "oos_recall"),
        ("Commands wrongly rejected as out-of-scope", "false_rejections"),
        ("Slot exactly right, end to end", "slot_end_to_end"),
        ("Slot exactly right, given the right intent", "slot_given_intent"),
        ("Rejection threshold (from CV on train)", "threshold"),
        ("Latency per command, ms", "latency_ms"),
    ]

    def fmt(r, key):
        if isinstance(key, tuple):
            return " / ".join(fmt(r, k) for k in key)
        v = r[key]
        if key in ("threshold", "latency_ms"):
            return f"{v:g}"
        return "—" if v is None else f"{100 * v:.1f}%"

    lines = ["| Metric | " + " | ".join(r["embedder"] for r in reports) + " |",
             "|---|" + "---|" * len(reports)]
    lines += [f"| {label} | " + " | ".join(fmt(r, key) for r in reports) + " |" for label, key in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", choices=EMBEDDERS, action="append")
    parser.add_argument("--device", default=None, help="for the sentence encoders: cpu or cuda")
    args = parser.parse_args()
    reports = [evaluate(name, args.device) for name in args.embedder or EMBEDDERS]
    print(table(reports))


if __name__ == "__main__":
    main()
