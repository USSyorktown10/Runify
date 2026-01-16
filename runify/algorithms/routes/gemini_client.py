"""Lightweight wrapper for Google Gemini text-gen API.

This module provides safe helpers that will no-op if no GEMINI_API_KEY
is present in the environment. It uses a minimal HTTP request so there
are no heavy deps. Adapt the endpoint and payload to your Gemini plan.
"""
import os
import json
import urllib.request
import urllib.error
import time
from collections import deque

# Use the Google Generative Language quickstart endpoint by default.
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_ENDPOINT = os.environ.get(
    'GEMINI_ENDPOINT',
    'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent'
)

# Simple rate limiter: allow up to 14 requests per 60-second window to stay under 15/minute
REQUEST_WINDOW_SECONDS = 60
REQUEST_LIMIT = 14
_recent_calls = deque()

# Last raw response text from the most recent Gemini call (for audit)
LAST_RAW = None

# Simple token accounting: aim to stay under 250k tokens per 60s window.
# We use a conservative chars->tokens heuristic: 4 chars per token (so tokens ~= chars/4).
TOKEN_WINDOW_SECONDS = 60
TOKEN_LIMIT = 250000
_recent_token_usage = deque()  # tuples of (timestamp, token_count)
_recent_token_total = 0


def _extract_text_from_response(j):
    """Robustly extract the text string from a Gemini response JSON."""
    if not isinstance(j, dict):
        return None
    # candidates -> content -> parts -> text
    cands = j.get('candidates') or j.get('candidate')
    if isinstance(cands, list) and len(cands) > 0:
        first = cands[0]
        # try several common shapes
        if isinstance(first, dict):
            # content.parts[].text
            content = first.get('content') or first
            if isinstance(content, dict):
                parts = content.get('parts')
                if isinstance(parts, list) and parts:
                    texts = [p.get('text') for p in parts if isinstance(p, dict) and p.get('text')]
                    texts = [t for t in texts if isinstance(t, str) and t]
                    if texts:
                        return '\n'.join(texts)
                # fallback to content['text']
                if content.get('text'):
                    return content.get('text')
            # fallback to 'output' or 'text' on candidate
            if first.get('output'):
                return first.get('output')
            if first.get('text'):
                return first.get('text')
    # top-level text or output
    if j.get('output'):
        return j.get('output')
    if j.get('text'):
        return j.get('text')
    return None


def _call_gemini(prompt, timeout=10):
    """Call Gemini: prefer official SDK (google.genai) when available, else use HTTP quickstart.
    Returns generated text or None on failure.
    """
    global LAST_RAW, _recent_token_total, _recent_token_usage

    # rate limiting: ensure we don't exceed REQUEST_LIMIT per REQUEST_WINDOW_SECONDS
    try:
        now = time.monotonic()
        # prune old timestamps
        while _recent_calls and (now - _recent_calls[0]) > REQUEST_WINDOW_SECONDS:
            _recent_calls.popleft()
        if len(_recent_calls) >= REQUEST_LIMIT:
            # need to wait until oldest timestamp is older than window
            wait_for = REQUEST_WINDOW_SECONDS - (now - _recent_calls[0]) + 0.1
            time.sleep(wait_for)
        # record this call timestamp
        _recent_calls.append(time.monotonic())
    except Exception:
        # if anything goes wrong, do not prevent the call
        pass

    # token accounting: approximate tokens for the prompt and throttle if needed
    try:
        now = time.monotonic()
        # prune old token records
        while _recent_token_usage and (now - _recent_token_usage[0][0]) > TOKEN_WINDOW_SECONDS:
            ts, tk = _recent_token_usage.popleft()
            _recent_token_total -= tk
        # estimate prompt tokens conservatively
        prompt_chars = len(prompt)
        est_prompt_tokens = int(prompt_chars / 4) + 1
        # if sending prompt would exceed TOKEN_LIMIT, wait until window frees
        if _recent_token_total + est_prompt_tokens >= TOKEN_LIMIT:
            # compute time to wait until enough tokens fall out of window
            if _recent_token_usage:
                oldest_ts = _recent_token_usage[0][0]
                wait_for = TOKEN_WINDOW_SECONDS - (now - oldest_ts) + 0.1
                time.sleep(wait_for)
                # prune again after sleeping
                now = time.monotonic()
                while _recent_token_usage and (now - _recent_token_usage[0][0]) > TOKEN_WINDOW_SECONDS:
                    ts, tk = _recent_token_usage.popleft()
                    _recent_token_total -= tk
        # record prompt tokens preemptively (they'll be included in the window)
        _recent_token_usage.append((time.monotonic(), est_prompt_tokens))
        _recent_token_total += est_prompt_tokens
    except Exception:
        pass

    # SDK path
    try:
        import google.genai as genai
        # create client (uses environment auth if available)
        client = genai.Client()
        # Use model gemini-2.0-flash-lite per user request
        try:
            resp = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
            print(resp)
        except Exception:
            # Some SDK builds expect contents wrapped; try list form
            resp = client.models.generate_content(model="gemini-2.5-flash-lite", contents=[prompt])
            print(resp)
        # Try to extract text from SDK response object
        try:
            # resp.candidates[0].content.parts -> each part has .text
            cands = getattr(resp, 'candidates', None) or getattr(resp, 'candidates', None)
            if cands and len(cands) > 0:
                first = cands[0]
                content = getattr(first, 'content', None) or first
                # content.parts
                parts = getattr(content, 'parts', None)
                if parts:
                    texts = []
                    for p in parts:
                        t = getattr(p, 'text', None) or (p.get('text') if isinstance(p, dict) else None)
                        if t:
                            texts.append(t)
                    if texts:
                                out_text = '\n'.join(texts)
                                LAST_RAW = out_text
                                # account for response tokens conservatively
                                try:
                                    resp_chars = len(out_text)
                                    resp_tokens = int(resp_chars / 4) + 1
                                    _recent_token_usage.append((time.monotonic(), resp_tokens))
                                    _recent_token_total += resp_tokens
                                except Exception:
                                    pass
                                return out_text
                # fallback to candidate text
                t = getattr(first, 'text', None) or (first.get('text') if isinstance(first, dict) else None)
                if t:
                    LAST_RAW = t
                    try:
                        resp_chars = len(t)
                        resp_tokens = int(resp_chars / 4) + 1
                        _recent_token_usage.append((time.monotonic(), resp_tokens))
                        _recent_token_total += resp_tokens
                    except Exception:
                        pass
                    return t
        except Exception:
            pass
    except Exception:
        # SDK not available or failed; fall through to HTTP quickstart
        pass

    # HTTP quickstart fallback
    if not GEMINI_API_KEY:
        return None
    payload = {
        'contents': [
            {
                'parts': [
                    {'text': prompt}
                ]
            }
        ]
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(GEMINI_ENDPOINT, data=data, method='POST')
    req.add_header('Content-Type', 'application/json')
    req.add_header('X-goog-api-key', GEMINI_API_KEY)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode('utf-8')
            try:
                j = json.loads(body)
                out = _extract_text_from_response(j) or body
                LAST_RAW = out
                try:
                    resp_chars = len(out)
                    resp_tokens = int(resp_chars / 4) + 1
                    _recent_token_usage.append((time.monotonic(), resp_tokens))
                    _recent_token_total += resp_tokens
                except Exception:
                    pass
                return out
            except Exception:
                LAST_RAW = body
                try:
                    resp_chars = len(body)
                    resp_tokens = int(resp_chars / 4) + 1
                    _recent_token_usage.append((time.monotonic(), resp_tokens))
                    _recent_token_total += resp_tokens
                except Exception:
                    pass
                return body
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode('utf-8')
            LAST_RAW = body
            try:
                resp_chars = len(body)
                resp_tokens = int(resp_chars / 4) + 1
                _recent_token_usage.append((time.monotonic(), resp_tokens))
                _recent_token_total += resp_tokens
            except Exception:
                pass
            return body
        except Exception:
            return None
    except Exception:
        return None


def suggest_merge_group(runs_summary):
    """Ask Gemini whether runs in a group should be merged and suggest parameters.

    runs_summary: a short JSON-serializable summary of run centroids/lengths.
    Returns: dict or None. Example: {'merge': True, 'merge_radius_m': 20}
    """
    prompt = (
        "Given these runs (JSON):\n"
        f"{json.dumps(runs_summary)}\n\n"
        "Decide if these should be merged into a single canonical trail. "
        "Reply only with a JSON object: {'merge': true/false, 'merge_radius_m': <meters>}"
    )
    out = _call_gemini(prompt)
    if not out:
        return None

    # Try to find a balanced JSON object in the output (handles fenced code blocks)
    def _extract_json(s):
        start = s.find('{')
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(s)):
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
        return None

    try:
        cand = _extract_json(out)
        if cand:
            return json.loads(cand)
    except Exception:
        pass
    return None


def analyze_skeleton(skeleton_points):
    """Ask Gemini to summarize skeleton quality and potential issues.

    Returns a short dict describing issues or notes. No-op returns None.
    """
    prompt = (
        "Analyze these skeleton points (list of [lat,lng,elev]). "
        "Return a JSON object with keys: quality ('good'|'ok'|'poor'), notes (string).\n"
        f"Data: {json.dumps(skeleton_points[:200])}"
    )
    out = _call_gemini(prompt)
    if not out:
        return None

    def _extract_json(s):
        start = s.find('{')
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(s)):
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
        return None

    try:
        cand = _extract_json(out)
        if cand:
            return json.loads(cand)
    except Exception:
        return {'quality': 'unknown', 'notes': out[:400]}
    return None
