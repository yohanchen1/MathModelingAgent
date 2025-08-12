import os
import hashlib
import pickle

CACHE_DIR = ".cache"

def setup_cache():
    """Creates the cache directory if it doesn't exist."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def get_cache_key(key_data):
    """Generates a SHA256 hash for the given data to use as a cache key."""
    if isinstance(key_data, str):
        key_data = key_data.encode('utf-8')
    return hashlib.sha256(key_data).hexdigest()

def get_cached_data(key):
    """Retrieves data from the cache."""
    cache_file = os.path.join(CACHE_DIR, key)
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            try:
                return pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                # Cache file is corrupted or empty, treat as a cache miss
                return None
    return None

def set_cached_data(key, data):
    """Saves data to the cache."""
    cache_file = os.path.join(CACHE_DIR, key)
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

def get_file_cache_key(filepath):
    """Generates a cache key for a file based on its path and modification time."""
    try:
        mod_time = os.path.getmtime(filepath)
        return get_cache_key(f"{filepath}{mod_time}")
    except FileNotFoundError:
        return None
