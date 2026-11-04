# Helpers for demos/<name>.setup.sh, sourced inside the demo workspace.
PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO"

# cfg '<python>' — edit config/agent.yaml in place; the code sees it as `cfg`.
cfg() {
    "$PY" - "$1" <<'EOF'
import sys, yaml
path = "config/agent.yaml"
cfg = yaml.safe_load(open(path))
exec(sys.argv[1])
yaml.safe_dump(cfg, open(path, "w"), allow_unicode=True, sort_keys=False)
EOF
}

# The `context` node's params dict, for cfg snippets.
CTX='next(n["context"] for n in cfg["pipeline"] if "context" in n)'

# rag_docs — copy the fixture documents (with their dates) and index them now,
# so the recording doesn't wait for the background ingestion.
rag_docs() {
    cp "$REPO"/demos/fixtures/rag/*.md .docs/
    touch -d '2025-02-10 12:00' .docs/notes-de-frais-2025.md
    touch -d '2026-09-15 12:00' .docs/notes-de-frais-2026.md
    touch -d '2026-02-20 12:00' .docs/teletravail.md
    touch -d '2026-01-05 12:00' .docs/onboarding.md
    "$PY" -c "import asyncio; from storage.ingest import ingest; asyncio.run(ingest('.docs/'))"
}
