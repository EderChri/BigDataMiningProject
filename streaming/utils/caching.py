import pickle
from pathlib import Path
import hashlib
import json
from typing import Dict, List


def get_cache_key(data_dir: str, split: str, max_messages: int, all_messages: bool,
                  exclude_duplicates: bool, update_interval: int) -> str:
    """Generate unique cache key from parameters"""
    params = f"{data_dir}_{split}_{max_messages}_{all_messages}_{exclude_duplicates}_{update_interval}"
    return hashlib.md5(params.encode()).hexdigest()

def load_cached_results(cache_file: Path) -> Dict | None:
    """Load results from cache file if exists"""
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def save_results(cache_file: Path, summary: Dict, snapshots: List[Dict]) -> None:
    """Save results to cache"""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump({'summary': summary, 'snapshots': snapshots}, f)
