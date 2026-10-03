#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$(pwd)/external/visual-cad-public}"
mkdir -p "$ROOT"
clone_or_update() {
  local url="$1" name="$2"
  if [[ -d "$ROOT/$name/.git" ]]; then
    git -C "$ROOT/$name" pull --ff-only
  else
    git clone --depth 1 "$url" "$ROOT/$name"
  fi
}
clone_or_update https://github.com/D4V1ND/tech_europe.git drawing2cad
clone_or_update https://github.com/skykim/cad-image-to-3d.git cad-image-to-3d
clone_or_update https://github.com/lllssc/Drawing2CAD.git Drawing2CAD-research
printf 'Public visual-CAD references are in %s\n' "$ROOT"
