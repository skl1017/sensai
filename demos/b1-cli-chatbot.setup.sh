# Two broken configs to show clean error handling (model missing, server down).
sed 's/^  model: .*/  model: modele-introuvable/' config/agent.yaml >config/modele-introuvable.yaml
sed 's#^  host: .*#  host: http://localhost:9999#' config/agent.yaml >config/ollama-hors-ligne.yaml
