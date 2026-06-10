#!/usr/bin/env bash
# gen_develop_delta.sh — regenerate the AUTO block in UPSTREAM.md §2.5 with the exact set of
# commits and files on `develop` that are NOT on `main`.
#
# This is the DETERMINISTIC core of the auto-delta workflow (the "hybrid" engine). It needs no
# LLM and no credentials — it just renders `git log main..develop` + `git diff` into a markdown
# table between the <!-- BEGIN/END AUTO:develop-delta --> markers in UPSTREAM.md. An optional
# Claude pass (in the workflow) may afterwards enrich the prose; this script is always correct on
# its own and is safe to run locally.
#
# Usage:
#   bash upstream-sync/gen_develop_delta.sh            # uses origin/main..HEAD (or develop)
#   BASE=origin/main HEAD_REF=origin/develop bash upstream-sync/gen_develop_delta.sh
#
# Exit 0 always (idempotent); writes UPSTREAM.md in place. Prints "CHANGED" or "UNCHANGED".
set -euo pipefail

LEDGER="${LEDGER:-UPSTREAM.md}"
BASE="${BASE:-origin/main}"
HEAD_REF="${HEAD_REF:-HEAD}"
REPO_SLUG="${REPO_SLUG:-fl97inc/SkyRL}"
BEGIN="<!-- BEGIN AUTO:develop-delta -->"
END="<!-- END AUTO:develop-delta -->"

cd "$(git rev-parse --show-toplevel)"

if [[ ! -f "$LEDGER" ]]; then
  echo "ERROR: $LEDGER not found at repo root" >&2
  exit 1
fi
if ! grep -qF "$BEGIN" "$LEDGER" || ! grep -qF "$END" "$LEDGER"; then
  echo "ERROR: AUTO markers not found in $LEDGER — refusing to guess insertion point" >&2
  exit 1
fi

# Resolve refs defensively (CI may name them differently); fall back to local branch names.
resolve() { git rev-parse --verify --quiet "$1^{commit}" >/dev/null 2>&1 && echo "$1"; }
BASE_REF="$(resolve "$BASE" || resolve "${BASE#origin/}" || echo "$BASE")"
TIP_REF="$(resolve "$HEAD_REF" || resolve "${HEAD_REF#origin/}" || echo "$HEAD_REF")"

base_sha="$(git rev-parse --short "$BASE_REF")"
tip_sha="$(git rev-parse --short "$TIP_REF")"
mb="$(git merge-base "$BASE_REF" "$TIP_REF")"
mb_short="$(git rev-parse --short "$mb")"

# Number of commits the delta covers (merge-base..tip, excluding base's own history).
n_commits="$(git rev-list --count "$BASE_REF..$TIP_REF")"

# Build the generated body into a temp file.
body="$(mktemp)"
{
  echo "_Last generated for \`$BASE_REF\` ($base_sha) ↔ \`$TIP_REF\` ($tip_sha); merge-base \`$mb_short\`. **Auto-generated — do not edit by hand.**_"
  echo

  if [[ "$n_commits" -eq 0 ]]; then
    echo "\`develop\` carries **no** commits beyond \`main\` — it is currently a clean mirror."
  else
    echo "\`develop\` is **$n_commits commit(s)** ahead of \`main\`."
    echo
    echo "### Commits on \`develop\` not on \`main\`"
    echo
    echo "| commit | subject | author | date |"
    echo "|---|---|---|---|"
    # Subjects are truncated to 60 cols; any literal pipe in a subject is escaped so it can't
    # break the markdown table.
    git log --no-merges --date=short \
      --pretty=format:"%h%x09%s%x09%an%x09%ad" "$BASE_REF..$TIP_REF" \
      | awk -F'\t' '{gsub(/\|/,"\\|",$2); printf "| `%s` | %s | %s | %s |\n", $1, substr($2,1,60), $3, $4}' || true
    echo
    merges="$(git log --merges --date=short --pretty=format:"| \`%h\` | %<(60,trunc)%s | %an | %ad |" "$BASE_REF..$TIP_REF" || true)"
    if [[ -n "$merges" ]]; then
      echo "<details><summary>merge commits ($(git rev-list --count --merges "$BASE_REF..$TIP_REF"))</summary>"
      echo
      echo "| commit | subject | author | date |"
      echo "|---|---|---|---|"
      echo "$merges"
      echo
      echo "</details>"
    fi
    echo
    echo "### Files changed (\`$BASE_REF...$TIP_REF\`)"
    echo
    echo '```'
    git diff --stat "$BASE_REF...$TIP_REF" | tail -n +1
    echo '```'
  fi
} > "$body"

# Splice the body between the markers, preserving everything outside (incl. §2 curation).
out="$(mktemp)"
awk -v begin="$BEGIN" -v end="$END" -v bodyfile="$body" '
  $0 ~ begin {print; while ((getline line < bodyfile) > 0) print line; skip=1; next}
  $0 ~ end {skip=0}
  !skip {print}
' "$LEDGER" > "$out"

if cmp -s "$LEDGER" "$out"; then
  rm -f "$out" "$body"
  echo "UNCHANGED"
else
  mv "$out" "$LEDGER"
  rm -f "$body"
  echo "CHANGED"
fi
