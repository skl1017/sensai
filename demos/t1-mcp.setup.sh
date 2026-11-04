# Two MCP servers: the official filesystem one, and our own (serves the calculator).
source "$REPO/demos/_lib.sh"
mkdir -p notes
printf 'Liste de courses :\n- farine\n- lait\n- beurre\n' > notes/courses.txt
cfg "cfg['tools'] = [t for t in cfg['tools'] if t not in ('calculator', 'read_file', 'list_dir', 'write_file')]
cfg['mcp_servers'] = [
    {'name': 'filesystem', 'command': 'npx',
     'args': ['-y', '@modelcontextprotocol/server-filesystem', '.'],
     'read_only': ['read_text_file', 'read_file', 'list_directory', 'directory_tree', 'search_files', 'get_file_info', 'list_allowed_directories']},
    {'name': 'sensai', 'command': '$PY', 'args': ['-m', 'mcp_server.server'],
     'env': {'PYTHONPATH': '$REPO'}, 'read_only': ['calculator']},
]"
