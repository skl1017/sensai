# A tiny context window so compression triggers after a few turns.
source "$REPO/demos/_lib.sh"
cfg "cfg['llm']['max_context_tokens'] = 1400
$CTX['budget'] = {'reserve_output': 512, 'trigger_ratio': 0.6, 'keep_last_turns': 2}
cfg['tools'] = [t for t in cfg['tools'] if t != 'memory']
$CTX['sections'] = ['persona', 'profile']"
