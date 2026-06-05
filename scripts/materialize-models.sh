#!/usr/bin/env bash
# Matérialise les poids LFS (pointeurs → vrais fichiers) pour l'inférence.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VIDEOMAE_WEIGHTS="outputs/models/videomae_soccernet/best_model/model.safetensors"

is_lfs_pointer() {
  local f="$1"
  [ -f "$f" ] && [ "$(wc -c < "$f")" -lt 1024 ] && head -1 "$f" | grep -q "git-lfs"
}

materialize_videomae() {
  if [ ! -f "$VIDEOMAE_WEIGHTS" ]; then
    return 0
  fi
  if ! is_lfs_pointer "$VIDEOMAE_WEIGHTS"; then
    return 0
  fi

  echo "→ Poids VideoMAE : pointeur LFS détecté, matérialisation…"
  git lfs checkout "$VIDEOMAE_WEIGHTS" 2>/dev/null || true

  if is_lfs_pointer "$VIDEOMAE_WEIGHTS"; then
    oid="$(sed -n 's/^oid sha256://p' "$VIDEOMAE_WEIGHTS")"
    cache=".git/lfs/objects/${oid:0:2}/${oid:2:2}/$oid"
    if [ -f "$cache" ]; then
      cp "$cache" "$VIDEOMAE_WEIGHTS"
      echo "   Copié depuis le cache LFS local ($(du -h "$VIDEOMAE_WEIGHTS" | cut -f1))"
    else
      echo "   Échec — lancez : git lfs pull"
      return 1
    fi
  else
    echo "   OK ($(du -h "$VIDEOMAE_WEIGHTS" | cut -f1))"
  fi
}

materialize_videomae
