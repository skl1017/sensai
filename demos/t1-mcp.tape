# T1 — MCP: a preexisting server (filesystem) and our own (sensai, serves the calculator).
Output demos/t1-mcp.gif
Output demos/t1-mcp.mp4
Source demos/_settings.tape

Type "grep -A3 '^- name\|^  name' config/agent.yaml | grep -A2 'name:'"
Enter
Sleep 3s
Type "clear"
Enter
Type "sensai"
Enter
Wait /^>\s*$/
Type "/tools"
Enter
Sleep 4s
Type "Combien font 1234 * 5678 ?"
Enter
Wait /^>\s*$/
Sleep 1s
Type "Lis le fichier notes/courses.txt et dis-moi ce qu'il manque pour faire des crêpes."
Enter
Wait /^>\s*$/
Sleep 4s
Type "/quit"
Enter
Sleep 1s
