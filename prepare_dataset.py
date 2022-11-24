import argparse
from preprocess.dataset import BeatsRhythmsDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Prepare Dataset')
    parser.add_argument('--mono', type=bool, default=True)
    args = parser.parse_args()
    
    mono = args.mono
    dataset = BeatsRhythmsDataset(seq_len = 64) #seq_len = 64 is not used
    dataset.load(mono = mono, force_prepare = True)
    dataset.save_processed_to_cache()

    
    