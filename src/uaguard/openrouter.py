import hashlib
import json
import os
import time
import urllib.error
import urllib.request

from .config import CACHE

URL = "https://openrouter.ai/api/v1/chat/completions"


def chat(model: str, system: str, user: str, max_tokens: int = 1024) -> dict:
    """One deterministic chat completion, cached on disk by request content."""
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    path = CACHE / "openrouter" / f"{digest}.json"
    if path.exists():
        return json.loads(path.read_text())
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"}
    for attempt in range(5):
        try:
            request = urllib.request.Request(URL, json.dumps(body).encode(), headers)
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.load(response)
            payload["latency_s"] = time.perf_counter() - started
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    if "choices" not in payload:
        raise RuntimeError(f"OpenRouter error: {payload.get('error')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False))
    return payload
