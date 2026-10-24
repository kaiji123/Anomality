from __future__ import annotations

import json
import os
from pathlib import Path


def generate_commentary(record: dict, cache_dir: str | Path = "results/narration") -> dict[str, str]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{record.get('event_id', 'record')}.json"
    if out.exists():
        return json.loads(out.read_text())
    if not os.getenv("GEMINI_API_KEY"):
        payload = {"headline": f"Event {record.get('event_id', 'record')} summary", "explanation": f"{record.get('event_type', 'event')} for {record.get('asset', 'asset')}: {record.get('reason', 'n/a')}", "source": "template"}
        out.write_text(json.dumps(payload))
        return payload
    payload = {"headline": f"Event {record.get('event_id', 'record')} summary", "explanation": f"{record.get('event_type', 'event')} for {record.get('asset', 'asset')}: {record.get('reason', 'n/a')}", "source": "template"}
    out.write_text(json.dumps(payload))
    return payload
