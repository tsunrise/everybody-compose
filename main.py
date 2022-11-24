import argparse
from utils.model import train
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train DeepBeats')
    parser.add_argument('-m','--model_name', type=str)
    parser.add_argument('-nf','--n_files', type=int, default=-1)
    parser.add_argument('-n','--n_epochs', type=int, default=100)
    parser.add_argument('-d','--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-s','--snapshots_freq', type=int, default=100)
    parser.add_argument('-c','--checkpoint', type=str, default=None)
    args = parser.parse_args()

    main_args = parser.parse_args()
    train(**vars(main_args))
