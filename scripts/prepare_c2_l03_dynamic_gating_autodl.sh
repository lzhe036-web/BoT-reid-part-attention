#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

REMOTE_URL="https://github.com/lzhe036-web/BoT-reid-part-attention.git"
DYNAMIC_BRANCH="exp/c2-l03-multi-granularity-dynamic-gating"
STATIC_BRANCH="exp/c2-l03-multi-granularity-local-feature"
STATIC_SHA="9cd7dbcee07b255803c8c21f4d9c5ee67a30930e"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

verify_remote_static() {
  local remote="$1"
  local rows=()
  local sha=""
  local ref=""
  local extra=""
  mapfile -t rows < <(git ls-remote --heads "$remote" "$STATIC_BRANCH")
  [[ "${#rows[@]}" -eq 1 ]] || fail "Static baseline remote lookup must return exactly one row."
  read -r sha ref extra <<<"${rows[0]}"
  [[ -z "$extra" ]] || fail "Static baseline remote lookup returned malformed evidence."
  [[ "$sha" == "$STATIC_SHA" ]] || fail "Static baseline remote SHA mismatch: $sha"
  [[ "$ref" == "refs/heads/${STATIC_BRANCH}" ]] || fail "Static baseline remote ref mismatch: $ref"
  printf '%s\n' "$sha"
}

CURRENT_BRANCH="$(git branch --show-current)"
[[ "$CURRENT_BRANCH" == "$DYNAMIC_BRANCH" ]] || fail "Expected branch $DYNAMIC_BRANCH, got $CURRENT_BRANCH"

DYNAMIC_HEAD_BEFORE="$(git rev-parse HEAD)"
[[ -n "$DYNAMIC_HEAD_BEFORE" ]] || fail "Dynamic HEAD is not resolvable."
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || fail "Worktree must be clean before AutoDL preparation."

git config --local http.version HTTP/1.1
git config --local core.compression 0
git config --local http.lowSpeedLimit 1
git config --local http.lowSpeedTime 30
git remote set-url origin "$REMOTE_URL"
[[ "$(git remote get-url origin)" == "$REMOTE_URL" ]] || fail "origin is not the canonical GitHub URL."

REMOTE_STATIC_SHA="$(verify_remote_static "$REMOTE_URL")"
[[ "$REMOTE_STATIC_SHA" == "$STATIC_SHA" ]] || fail "Canonical remote verification failed."

if ! git cat-file -e "${STATIC_SHA}^{commit}" 2>/dev/null; then
  fail "Static baseline commit object missing; deepen/fetch history first."
fi

git branch -f "$STATIC_BRANCH" "$STATIC_SHA"
git update-ref "refs/remotes/origin/${STATIC_BRANCH}" "$STATIC_SHA"

[[ "$(git rev-parse "$STATIC_BRANCH")" == "$STATIC_SHA" ]] || fail "Local Static compatibility ref mismatch."
[[ "$(git rev-parse "origin/${STATIC_BRANCH}")" == "$STATIC_SHA" ]] || fail "origin/Static compatibility ref mismatch."
[[ "$(git merge-base "$STATIC_BRANCH" HEAD)" == "$STATIC_SHA" ]] || fail "Dynamic/Static merge-base mismatch."

FINAL_REMOTE_STATIC_SHA="$(verify_remote_static origin)"
[[ "$FINAL_REMOTE_STATIC_SHA" == "$STATIC_SHA" ]] || fail "Final origin verification failed."
[[ "$(git rev-parse HEAD)" == "$DYNAMIC_HEAD_BEFORE" ]] || fail "Dynamic HEAD changed during AutoDL preparation."
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || fail "Worktree changed during AutoDL preparation."

printf 'AutoDL Dynamic Gating evidence preflight passed.\n'
printf 'Dynamic HEAD: %s\n' "$DYNAMIC_HEAD_BEFORE"
printf 'Static remote/local/tracking/merge-base SHA: %s\n' "$STATIC_SHA"
