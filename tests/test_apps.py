from aiforme.voice_orchestrator.apps import best_match, clean_name, transliterate, variants

INSTALLED = ["Telegram", "Google Chrome", "Steam", "Discord", "Visual Studio Code", "Калькулятор",
             "Spotify", "OBS Studio", "Dota 2", "Блокнот"]


def match(phrase):
    found = best_match(phrase, INSTALLED)
    return found[0] if found else None


def test_transliteration_and_variants():
    assert transliterate("телеграм") == "telegram"
    assert "discord" in variants("дискорда")       # alias after the case ending is dropped
    assert "chrome" in variants("хром")


def test_clean_name():
    assert clean_name("WINWORD.EXE") == "winword"
    assert clean_name("Telegram Desktop®") == "telegram desktop"


def test_spoken_names():
    assert match("телеграм") == "Telegram"
    assert match("телегу") == "Telegram"
    assert match("хром") == "Google Chrome"
    assert match("Chrome") == "Google Chrome"
    assert match("вс код") == "Visual Studio Code"
    assert match("калькулятор") == "Калькулятор"
    assert match("спотифай") == "Spotify"
    assert match("дота") == "Dota 2"


def test_leftover_command_words_do_not_hide_the_app():
    assert match("а можешь запустить стим") == "Steam"
    assert match("could you launch chrome") == "Google Chrome"


def test_unknown_apps_are_not_guessed():
    assert match("фотошоп") is None
    assert match("окно в комнате") is None
