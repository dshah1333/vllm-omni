"""In-process tap of the thinker's engine-accepted text tokens.

Public vLLM 0.27 re-presents resumable duplex wakes as prefills, so the
connector's new_token_ids snapshot returns stale placeholder tokens and
request.output_token_ids is cleared by resumable cleanup. The scheduler
records each accepted step's token here; the thinker->talker producer
rebuilds the text timeline from it.
"""
_TOKENS: dict[str, list[int]] = {}


def record(request_id, token_id):
    _TOKENS.setdefault(str(request_id), []).append(int(token_id))


def get(request_id):
    return list(_TOKENS.get(str(request_id), ()))
