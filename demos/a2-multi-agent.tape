# A2 — Multi-agent: the main agent delegates math to a tutor sub-agent (its own persona, model and tools).
Output demos/a2-multi-agent.gif
Output demos/a2-multi-agent.mp4
Source demos/_settings.tape

Type "grep -v '^#' config/agents/tutor.yaml | grep -v '^$'"
Enter
Sleep 3s
Type "clear"
Enter
Type "sensai"
Enter
Wait /^>\s*$/
Type "/tools"
Enter
Sleep 3s
Type "Demande au tuteur de m'expliquer pas à pas comment calculer 15 % de 240."
Enter
Wait /^>\s*$/
Sleep 4s
Type "/quit"
Enter
Sleep 1s
