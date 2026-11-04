# T2 — Sandboxed code execution (Docker): the agent runs its code and reads the result.
Output demos/t2-code-sandbox.gif
Output demos/t2-code-sandbox.mp4
Source demos/_settings.tape

Type "sensai"
Enter
Wait /^>\s*$/
Type "Écris un script Python qui calcule les 20 premiers nombres premiers, exécute-le et donne le résultat."
Enter
Wait /^>\s*$/
Sleep 2s
# A traceback is an observation: the agent fixes its code.
Type "Exécute ce code tel quel, puis corrige-le s'il plante : print(sum(int(x) for x in '1,2,trois,4'.split(',')))"
Enter
Wait /^>\s*$/
Sleep 4s
Type "/quit"
Enter
Sleep 1s
