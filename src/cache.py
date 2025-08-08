"""
Cache utilities for JWST photometry pipeline (Phase 7 advanced caching).

Provides a simple, robust disk cache with namespaced keys and pluggable
serialization, plus a decorator for transparent function result caching.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


def _hash_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    """Create a stable hash for arbitrary JSON-serializable objects.

    Falls back to pickle for non-JSON-serializable objects.
    """
    try:
        payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return _hash_bytes(payload)


@dataclass
class CacheConfig:
    base_dir: Path
    namespace: str = "default"
    enabled: bool = True
    binary: bool = True  # pickle by default


class CacheManager:
    """
    Lightweight disk cache.

    Keys are composed as namespace:key_hash; values are stored as .pkl or .json.
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self.base_dir = Path(config.base_dir)
        self.ns_dir = self.base_dir / config.namespace
        if self.config.enabled:
            self.ns_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        key_hash = stable_hash(key)
        suffix = ".pkl" if self.config.binary else ".json"
        return self.ns_dir / f"{key_hash}{suffix}"

    def get(self, key: str) -> Optional[Any]:
        if not self.config.enabled:
            return None
        path = self._path_for_key(key)
        if not path.exists():
            return None
        try:
            if self.config.binary:
                with open(path, "rb") as f:
                    return pickle.load(f)
            else:
                with open(path, "r") as f:
                    return json.load(f)
        except Exception:
            # Corrupted cache entry—remove and miss
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            return None

    def set(self, key: str, value: Any) -> None:
        if not self.config.enabled:
            return
        path = self._path_for_key(key)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            if self.config.binary:
                with open(tmp, "wb") as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(tmp, "w") as f:
                    json.dump(value, f, default=str)
            os.replace(tmp, path)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass


def disk_cache(
    cache: CacheManager,
    key_builder: Optional[Callable[..., str]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to cache function results on disk.

    Example:
        cache = CacheManager(CacheConfig(Path(".cache"), namespace="photometry"))

        @disk_cache(cache, key_builder=lambda image_id, **k: f"{image_id}|{k}")
        def run_photometry(image_id: str, data: np.ndarray, **kwargs):
            ...
    """

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def _wrapper(*args, **kwargs):
            if not cache.config.enabled:
                return func(*args, **kwargs)

            if key_builder is None:
                cache_key = f"{func.__module__}.{func.__name__}:{stable_hash((args, kwargs))}"
            else:
                try:
                    cache_key = key_builder(*args, **kwargs)
                except Exception:
                    cache_key = f"{func.__module__}.{func.__name__}:{stable_hash((args, kwargs))}"

            hit = cache.get(cache_key)
            if hit is not None:
                return hit

            result = func(*args, **kwargs)
            try:
                cache.set(cache_key, result)
            except Exception:
                # Non-fatal if value isn't serializable
                pass
            return result

        return _wrapper

    return _decorator
