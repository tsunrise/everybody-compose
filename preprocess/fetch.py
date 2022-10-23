# Downloader for the midi files with cache
import os
from pathlib import Path
import requests
from tqdm import tqdm

from utils.data_paths import DataPaths

def download(filename: str, url: str) -> Path:
    """Download a zip file from a URL if it's not already in the cache."""
    paths = DataPaths()
    cache_path = paths.downloads_dir / filename
    if not os.path.exists(cache_path):
        with requests.get(url, stream=True) as r:
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            chunk_size = 1024
            with open(cache_path, "wb") as f:
                print(f"Downloading {filename} from {url}")
                progress = tqdm(total = total_size_in_bytes, unit = 'iB', unit_scale = True, colour="cyan")
                for chunk in r.iter_content(chunk_size=chunk_size):
                    progress.update(len(chunk))
                    f.write(chunk)
                progress.close()
    else:
        print("Using cached: ", filename)
    return cache_path




