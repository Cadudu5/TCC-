#!/bin/zsh
set -e

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"
LABELER_ENV=".venv-rotulador"

if [[ ! -x "$LABELER_ENV/bin/python" ]]; then
  echo "Preparando o ambiente do rotulador..."
  python3 -m venv "$LABELER_ENV"
  "$LABELER_ENV/bin/python" -m pip install --upgrade pip
  "$LABELER_ENV/bin/python" -m pip install -r requirements-rotulador.txt
fi

exec "$LABELER_ENV/bin/python" rotulador_lite.py "$@"
