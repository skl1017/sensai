# A session with known node ids, so the tape can /edit them.
mkdir -p sessions
cat > sessions/voyage.json <<'JSON'
{"session_id": "voyage", "created_at": "2026-11-04T09:00:00+00:00", "updated_at": "2026-11-04T09:02:00+00:00", "head": "a2000000",
 "nodes": {
  "u1000000": {"id": "u1000000", "parent_id": null, "role": "user", "content": "Propose-moi une destination pour une semaine de vacances en octobre.", "created_at": "2026-11-04T09:00:00+00:00"},
  "a1000000": {"id": "a1000000", "parent_id": "u1000000", "role": "assistant", "content": "Lisbonne : 22 °C en octobre, des quartiers à pied, et des billets abordables.", "created_at": "2026-11-04T09:00:05+00:00"},
  "u2000000": {"id": "u2000000", "parent_id": "a1000000", "role": "user", "content": "Donne-moi un programme pour le premier jour.", "created_at": "2026-11-04T09:02:00+00:00"},
  "a2000000": {"id": "a2000000", "parent_id": "u2000000", "role": "assistant", "content": "Matin à l'Alfama, déjeuner au Time Out Market, coucher de soleil au Miradouro da Senhora do Monte.", "created_at": "2026-11-04T09:02:05+00:00"}
 }}
JSON

# The model answers instead of asking which day we mean.
source "$REPO/demos/_lib.sh"
cfg "cfg['tools'] = [t for t in cfg['tools'] if t != 'ask_user']"
