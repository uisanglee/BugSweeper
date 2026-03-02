#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: bash scripts/create_private_repo.sh <github_token> <owner_or_org> <new_repo_name>"
  exit 1
fi

TOKEN="$1"
OWNER="$2"
REPO="$3"

api_base="https://api.github.com"

# Determine whether owner is user or org to choose endpoint
owner_type=$(curl -s -H "Authorization: token ${TOKEN}" "${api_base}/users/${OWNER}" | python - <<'PY'
import json,sys
try:
    data=json.load(sys.stdin)
    print(data.get('type',''))
except Exception:
    print('')
PY
)

if [ "${owner_type}" = "Organization" ]; then
  endpoint="${api_base}/orgs/${OWNER}/repos"
else
  endpoint="${api_base}/user/repos"
fi

payload=$(cat <<JSON
{
  "name": "${REPO}",
  "private": true,
  "auto_init": false
}
JSON
)

resp=$(curl -s -X POST \
  -H "Authorization: token ${TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  "${endpoint}" \
  -d "${payload}")

clone_url=$(echo "${resp}" | python - <<'PY'
import json,sys
try:
    data=json.load(sys.stdin)
    print(data.get('clone_url',''))
except Exception:
    print('')
PY
)

if [ -z "${clone_url}" ]; then
  echo "[ERROR] Repository creation failed. Raw response:"
  echo "${resp}"
  exit 1
fi

echo "[OK] Created private repository: ${clone_url}"

auth_url="https://${TOKEN}@github.com/${OWNER}/${REPO}.git"
echo "To push current branch:"
echo "  git remote add private-origin ${auth_url}"
echo "  git push -u private-origin HEAD"
