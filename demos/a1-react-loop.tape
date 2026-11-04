# A1 — ReAct loop: the model chains several tool calls before answering.
Output demos/a1-react-loop.gif
Output demos/a1-react-loop.mp4
Source demos/_settings.tape

Type "sensai"
Enter
Wait /^>\s*$/
Type "J'ai 1500 € placés à 4 % par an pendant 7 ans. Combien j'aurai à la fin, et combien chacun si on partage en 3 ?"
Enter
Wait /^>\s*$/
Sleep 4s
Type "/quit"
Enter
Sleep 1s
