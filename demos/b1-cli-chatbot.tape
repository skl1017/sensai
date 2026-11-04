# B1 — CLI chatbot: streaming, in-session history, clean errors, model from config.
Output demos/b1-cli-chatbot.gif
Output demos/b1-cli-chatbot.mp4
Source demos/_settings.tape

# The model comes from the config file, not the code.
Type "grep -n 'model:' config/agent.yaml"
Enter
Sleep 2s

Type "sensai"
Enter
Wait /^>\s*$/

# Streaming answer.
Type "Je m'appelle Alex. Explique en trois phrases ce qu'est un LLM."
Enter
Wait /^>\s*$/
Sleep 2s

# Empty input is ignored.
Enter
Sleep 1s

# The previous turn is in context.
Type "Comment je m'appelle ?"
Enter
Wait /^>\s*$/
Sleep 2s

Type "/quit"
Enter
Sleep 1s
Type "clear"
Enter

# Another config, a model that does not exist: a clear error, no traceback.
Type "sensai --config config/modele-introuvable.yaml"
Enter
Wait /^>\s*$/
Type "Bonjour"
Enter
Wait /^>\s*$/
Sleep 2s
Type "/quit"
Enter
Sleep 1s

# Ollama unreachable: same.
Type "sensai --config config/ollama-hors-ligne.yaml"
Enter
Wait /^>\s*$/
Type "Bonjour"
Enter
Wait /^>\s*$/
Sleep 3s
Type "/quit"
Enter
Sleep 1s
