"""In-process tap of the thinker's engine-accepted text tokens.

Public vLLM 0.27 re-presents resumable duplex wakes as prefills, so the
connector's new_token_ids snapshot returns stale placeholder tokens and
request.output_token_ids is cleared by resumable cleanup. The scheduler
records each accepted step's token here; the thinker->talker producer
rebuilds the text timeline from it.
"""
_TOKENS: dict[str, list[int]] = {}
# Duplex streams have no in-band end signal at this layer (is_finished marks
# resumable segments, not the stream), so the cache is size-bounded instead of
# lifecycle-evicted: oldest request first, far above any realistic concurrent
# session count.
_MAX_REQUESTS = 64


def record(request_id, token_id):
    key = str(request_id)
    if key not in _TOKENS and len(_TOKENS) >= _MAX_REQUESTS:
        _TOKENS.pop(next(iter(_TOKENS)))
    _TOKENS.setdefault(key, []).append(int(token_id))


def get(request_id):
    return list(_TOKENS.get(str(request_id), ()))
