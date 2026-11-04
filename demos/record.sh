#!/usr/bin/env bash
# Record one or more feature demos with VHS, each in a fresh isolated workspace.
#
#   demos/record.sh                  # every demos/*.tape
#   demos/record.sh b1-cli-chatbot   # only these
#
# Each demo runs in demos/.work/ (gitignored): a copy of config/, its own data.db,
# sessions/ and profile, so recordings never touch your real state. An optional
# demos/<name>.setup.sh runs inside the workspace first (seed documents, profile...).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEMOS="$REPO/demos"
WORK="$DEMOS/.work"

command -v vhs >/dev/null || { echo "vhs not found: see demos/README.md" >&2; exit 1; }

if [ "$#" -eq 0 ]; then
    set -- $(cd "$DEMOS" && ls *.tape | grep -v '^_' | sed 's/\.tape$//')
fi

failed=()
for name in "$@"; do
    tape="$DEMOS/$name.tape"
    [ -f "$tape" ] || { echo "no such demo: $name" >&2; exit 1; }

    rm -rf "$WORK"
    mkdir -p "$WORK/data" "$WORK/.docs"
    # Reranker weights are large: reuse the downloaded ones.
    [ -d "$REPO/data/models" ] && ln -s "$REPO/data/models" "$WORK/data/models"
    cp -r "$REPO/config" "$WORK/config"
    find "$WORK/config" -name __pycache__ -prune -exec rm -rf {} +
    # No MCP server by default (t1 adds its own): fewer tools to pick from, no startup logs.
    (cd "$WORK" && REPO="$REPO" bash -c 'source "$REPO/demos/_lib.sh"; cfg "cfg[\"mcp_servers\"] = []"')

    # Shell setup sourced (hidden) at the start of every tape.
    cat >"$WORK/.demorc" <<EOF
cd "$WORK"
export PYTHONPATH="$REPO"
export PS1='\$ '
alias sensai='"$REPO/.venv/bin/python" "$REPO/demos/sensai_demo.py"'
EOF

    if [ -f "$DEMOS/$name.setup.sh" ]; then
        (cd "$WORK" && REPO="$REPO" bash "$DEMOS/$name.setup.sh")
    fi

    echo "== $name"
    (cd "$REPO" && vhs "$tape") || failed+=("$name")
done

rm -rf "$WORK"
if [ "${#failed[@]}" -gt 0 ]; then
    echo "FAILED: ${failed[*]}" >&2
    exit 1
fi
