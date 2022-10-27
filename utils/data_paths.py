"""
Project Tree Structure
.project_data
|   downloads/
|   snapshots/
|   prepared_data/
|   midi_outputs/
|   tensorboard/
"""
from pathlib import Path

class DataPaths:
    def __init__(self):
        self.cache_dir = Path(".project_data")
        self.cache_dir.mkdir(exist_ok=True)
        self.downloads_dir = self.cache_dir / "downloads"
        self.downloads_dir.mkdir(exist_ok=True)
        self.snapshots_dir = self.cache_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)
        self.prepared_data_dir = self.cache_dir / "prepared_data"
        self.prepared_data_dir.mkdir(exist_ok=True)
        self.midi_outputs_dir = self.cache_dir / "midi_outputs"
        self.midi_outputs_dir.mkdir(exist_ok=True)
        self.tensorboard_dir = self.cache_dir / "tensorboard"
        self.tensorboard_dir.mkdir(exist_ok=True)