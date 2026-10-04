"""
Minimal disk-backed cache so repeated runs on the same claim/query don't
re-hit the search API or the LLM (both are rate-limited / quota-limited).

One JSON file per cache key under data/cache/ - deliberately simple (no
TTL, no eviction) since this is a research-workflow convenience, not a
production cache. Delete data/cache/ manually to force a refresh.
"""
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CACHE_DIR


def _cache_path(namespace: str, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"{namespace}__{digest}.json"


def get(namespace: str, key: str) -> Optional[dict]:
    path = _cache_path(namespace, key)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def set(namespace: str, key: str, value) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(namespace, key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    set("demo", "hello", {"ok": True})
    print(get("demo", "hello"))
    print(get("demo", "missing-key"))
