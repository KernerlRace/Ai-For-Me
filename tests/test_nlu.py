import pytest

from aiforme.voice_orchestrator.intents import INTENTS, LABELS, OUT_OF_SCOPE
from aiforme.voice_orchestrator.nlu import (
    TEST,
    TRAIN,
    IntentClassifier,
    SlotExtractor,
    TfidfEmbedder,
    language,
    load_examples,
    normalize,
)


@pytest.fixture(scope="module")
def clf():
    return IntentClassifier(TfidfEmbedder()).fit(load_examples(TRAIN))


def test_normalize():
    assert normalize("Открой Telegram, пожалуйста!") == "открой telegram пожалуйста"
    assert normalize("type: I'll be home — at eight") == "type i'll be home at eight"
    assert normalize("включи-ка «ёлку»") == "включи ка елку"
    assert normalize("'quoted'") == "quoted"


def test_language():
    assert language("Открой Telegram") == "ru"
    assert language("open telegram") == "en"


@pytest.mark.parametrize("path", [TRAIN, TEST])
def test_dataset_is_consistent(path):
    examples = load_examples(path)
    assert {e.intent for e in examples} == set(LABELS)
    for e in examples:
        slot = INTENTS.get(e.intent)
        assert set(e.slots) == ({slot} if slot else set()), e
        if slot:
            assert f" {normalize(e.slots[slot])} " in f" {normalize(e.text)} ", e


def test_train_and_test_do_not_overlap():
    train = {normalize(e.text) for e in load_examples(TRAIN)}
    assert not train & {normalize(e.text) for e in load_examples(TEST)}


def test_slots_from_learned_carriers():
    slots = SlotExtractor().fit(load_examples(TRAIN))
    assert slots.extract("open_app", "Открой мне Telegram, пожалуйста") == {"app": "telegram"}
    assert slots.extract("close_app", "close the photoshop please") == {"app": "photoshop"}
    assert slots.extract("web_search", "загугли в каком году пала византия") == {
        "query": "в каком году пала византия"}            # "в" belongs to the query here
    assert slots.extract("find_file", "where is my diploma") == {"query": "diploma"}   # vocabulary fallback
    assert slots.extract("dictate", "напиши мне позже") == {"text": "мне позже"}       # text keeps "мне"
    assert slots.extract("mute", "mute") == {}


def test_slot_must_be_a_phrase_of_the_text():
    from aiforme.voice_orchestrator.nlu import Example
    with pytest.raises(ValueError):
        SlotExtractor().fit([Example("open steam", "open_app", {"app": "ste"})])


def test_tfidf_baseline_recognises_clear_commands(clf):
    assert clf.predict_one("открой телеграм").intent == "open_app"
    assert clf.predict_one("Close Chrome.").slots == {"app": "chrome"}
    assert clf.predict_one("сделай громче").intent == "volume_up"
    assert clf.predict_one("закажи пиццу").intent == OUT_OF_SCOPE


def test_threshold_is_chosen_on_train_only(clf):
    assert 0.0 <= clf.threshold < 1.0
    assert set(clf.cv_report) == {"in_scope_accuracy", "oos_recall", "score"}


def test_char_cnn_trains_from_scratch():
    pytest.importorskip("torch")
    from aiforme.voice_orchestrator.nlu import CharCNNEmbedder
    clf = IntentClassifier(CharCNNEmbedder(epochs=30), threshold=0.0).fit(load_examples(TRAIN))
    assert clf.predict_one("открой телеграм").intent == "open_app"
    assert clf.embedder.encode(["громче"]).shape == (1, 256)


def test_rejected_command_keeps_its_best_guess():
    clf = IntentClassifier(TfidfEmbedder(), threshold=0.99).fit(load_examples(TRAIN))
    parse = clf.predict_one("открой телеграм")
    assert parse.intent == OUT_OF_SCOPE
    assert parse.best_guess == "open_app"
