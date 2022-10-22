from typing import IO, List, Optional, Tuple, Union
import music21
import numpy as np
import preprocess.fetch as fetch
from torch.utils.data import Dataset


    
class BeatsRhythmsDataset(Dataset):
    def __init__(self):
        ... # TODO