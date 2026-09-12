import atexit
from pathlib import Path
from typing import Any, Optional
from diskcache import Cache

CACHE_DIR = Path("data/cache")
_CACHE = None


def get_cache() -> Cache:
    global _CACHE
    if _CACHE is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE = Cache(directory=str(CACHE_DIR))
        atexit.register(close_cache)
    return _CACHE


def cache_get(key: str) -> Optional[Any]:
    return get_cache().get(key)


def cache_set(key: str, value: Any) -> None:
    cache = get_cache()
    cache.set(key, value)


def flush_cache() -> None:
    global _CACHE
    if _CACHE is not None:
        try:
            _CACHE.cull()
        except Exception:
            pass


def close_cache() -> None:
    global _CACHE
    if _CACHE is not None:
        try:
            _CACHE.close()
        except Exception:
            pass
        _CACHE = None
