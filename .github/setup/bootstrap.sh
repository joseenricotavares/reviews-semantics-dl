#!/usr/bin/env bash
# One-time repo bootstrap: pushes secrets from .secrets.env and applies branch-protection.json.
# Requires: the authenticated gh user to have admin rights on the repo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS_FILE="$SCRIPT_DIR/../.secrets.env"
PROTECTION_FILE="$SCRIPT_DIR/branch-protection.json"

if ! gh auth status >/dev/null 2>&1; then
  gh auth login
fi

REPO=$(gh repo view --json nameWithOwner --jq .nameWithOwner)

if [ ! -f "$SECRETS_FILE" ]; then
  echo "::error::$SECRETS_FILE not found. Copy .secrets.env.example to .secrets.env and fill in real values first." >&2
  exit 1
fi

gh secret set --env-file "$SECRETS_FILE" --repo "$REPO"
echo "Secrets from $SECRETS_FILE pushed to ${REPO}."

gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  "repos/${REPO}/branches/main/protection" \
  --input "$PROTECTION_FILE"
echo "Branch protection applied to ${REPO}@main from ${PROTECTION_FILE}."
