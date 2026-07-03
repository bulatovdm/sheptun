#!/bin/bash
set -euo pipefail
ROOT="${CLAUDE_PROJECT_DIR:-$PWD}"
INDEX="$ROOT/.kb/memory/memory.md"
[[ -f "$INDEX" ]] || exit 0
jq -Rn --arg c "$(cat "$INDEX")" \
  '{hookSpecificOutput: {hookEventName: "SessionStart", additionalContext: $c}}'
