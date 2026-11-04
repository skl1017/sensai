# T3 — Web search (SearXNG): live results injected into the context.
Output demos/t3-web-search.gif
Output demos/t3-web-search.mp4
Source demos/_settings.tape

Type "sensai"
Enter
Wait /^>\s*$/
Type "Cherche sur le web la dernière version stable de Python et sa date de sortie."
Enter
Wait /^>\s*$/
Sleep 4s
Type "/quit"
Enter
Sleep 1s
