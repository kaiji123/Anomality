from __future__ import annotations

import os
from pathlib import Path


def synthesize_voice(text: str, event_id: str, cache_dir: str | Path = "results/narration") -> str | None:
    if not os.getenv("ELEVENLABS_API_KEY") or not os.getenv("ELEVENLABS_VOICE_ID"):
        return None
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    out = p / f"{event_id}.mp3"
    return str(out) if out.exists() else None
