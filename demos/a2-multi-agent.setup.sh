# Expose the math tutor sub-agent as `ask_tutor`; the main agent loses its calculator.
source "$REPO/demos/_lib.sh"
cfg "cfg['tools'] = [t for t in cfg['tools'] if t != 'calculator']
cfg['agents'] = [{'name': 'tutor', 'config': 'agents/tutor.yaml',
    'description': 'Tuteur de mathématiques : explique un calcul ou une notion mathématique pas à pas.',
    'max_depth': 2, 'max_calls': 3, 'timeout': 120}]"
