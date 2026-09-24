DEV = docker compose -f docker-compose.dev.yml
S   ?= dev
Q   ?= Bonjour !

.PHONY: dev up down logs shell test chat chat-stream reset

dev:            ## lance la stack de dev (hot reload)
	$(DEV) up --build
up:             ## lance la prod
	docker compose up -d --build
down:
	$(DEV) down
logs:
	$(DEV) logs -f sensai
shell:          ## shell dans le conteneur
	$(DEV) exec sensai bash
test:
	$(DEV) exec sensai pytest -q
chat:           ## make chat Q="ta question" S=session
	@curl -s localhost:8000/chat -d '{"text": "$(Q)", "session_id": "$(S)"}' | python3 -m json.tool
chat-stream:    ## idem, en streaming
	@curl -sN localhost:8000/chat -d '{"text": "$(Q)", "session_id": "$(S)", "stream": true}'
reset:          ## repart de zéro (volumes Ollama compris)
	$(DEV) down -v
