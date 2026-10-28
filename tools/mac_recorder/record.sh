#!/bin/bash
# Run in Terminal as `bash record.sh` (drag the file into the window): `bash` needs no execute bit and no
# Gatekeeper approval, which a file sent through a messenger or unpacked from a zip made on Windows lacks.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$HOME/.amadeus-recorder"

if ! python3 -c "import sys; assert sys.version_info >= (3, 9)" 2>/dev/null; then
    echo "Нужен Python. Сейчас появится окно установки инструментов разработчика:"
    echo "нажми «Установить», дождись конца (5–15 минут) и запусти эту команду ещё раз."
    xcode-select --install 2>/dev/null || true
    exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
    echo "Первый запуск: ставлю всё нужное, это минута-две..."
    python3 -m venv "$VENV"
    "$VENV/bin/python" -m pip install --quiet --disable-pip-version-check --upgrade pip
fi
"$VENV/bin/python" -m pip install --quiet --disable-pip-version-check sounddevice numpy

# caffeinate: the Mac must not fall asleep in the middle of a ten-minute room recording
caffeinate -i "$VENV/bin/python" "$DIR/record_mac.py" "$@"
