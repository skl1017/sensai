# No approval prompt here (shown in a3): the fix-and-retry loop runs on its own.
source "$REPO/demos/_lib.sh"
cfg "cfg['approval'] = [t for t in cfg['approval'] if t != 'run_python']"
