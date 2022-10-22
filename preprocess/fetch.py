# Downloader for the midi files with cache
import os
import requests
from tqdm import tqdm

from preprocess.constants import CACHE_DIR

def download(filename: str, url: str) -> str:
    """Download a zip file from a URL if it's not already in the cache."""
    download_dir = os.path.join(CACHE_DIR, "downloads")
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    cache_path = os.path.join(download_dir, filename)
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




