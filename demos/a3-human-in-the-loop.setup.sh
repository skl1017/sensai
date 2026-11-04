# Without ask_user, a refusal ends the turn instead of a follow-up question.
source "$REPO/demos/_lib.sh"
cfg "cfg['tools'] = [t for t in cfg['tools'] if t != 'ask_user']"
