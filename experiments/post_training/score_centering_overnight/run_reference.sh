#!/usr/bin/env bash
set -euo pipefail

campaign_dir="$(cd "$(dirname "$0")" && pwd)"
reference_dir=/tmp/score-centering-authors
git clone https://github.com/martin-marek/score-centering.git "$reference_dir"
git -C "$reference_dir" checkout --detach 7c56e9ee2972aa57f446cf564de1a1658d14b321
uv --no-config venv --python /usr/bin/python3.13 /tmp/score-centering-reference-env
uv --no-config pip install --python /tmp/score-centering-reference-env/bin/python --require-hashes -r "$campaign_dir/reference-requirements.lock"
exec /tmp/score-centering-reference-env/bin/python "$campaign_dir/reference.py" \
  --source "$reference_dir" --lock "$campaign_dir/reference-requirements.lock" "$@"
