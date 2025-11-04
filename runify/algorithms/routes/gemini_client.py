"""Lightweight wrapper for Google Gemini text-gen API.

This module provides safe helpers that will no-op if no GEMINI_API_KEY
is present in the environment. It uses a minimal HTTP request so there
are no heavy deps. Adapt the endpoint and payload to your Gemini plan.
"""
import os
import json
import urllib.request
import urllib.error

# Use the Google Generative Language quickstart endpoint by default.
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_ENDPOINT = os.environ.get(
    'GEMINI_ENDPOINT',
    'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent'
)


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
    # SDK path
    try:
        import google.genai as genai
        # create client (uses environment auth if available)
        client = genai.Client()
        # Use model gemini-2.0-flash-lite per user request
        try:
            resp = client.models.generate_content(model="gemini-2.0-flash-lite", contents=prompt)
            print(resp)
        except Exception:
            # Some SDK builds expect contents wrapped; try list form
            resp = client.models.generate_content(model="gemini-2.0-flash-lite", contents=[prompt])
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
                        return '\n'.join(texts)
                # fallback to candidate text
                t = getattr(first, 'text', None) or (first.get('text') if isinstance(first, dict) else None)
                if t:
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
                return _extract_text_from_response(j) or body
            except Exception:
                return body
    except urllib.error.HTTPError as e:
        try:
            return e.read().decode('utf-8')
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
