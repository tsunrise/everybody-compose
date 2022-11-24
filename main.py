import argparse
from utils.model import train
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train DeepBeats')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--n_files', type=int, default=-1)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--snapshots_freq', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    main_args = parser.parse_args()
    train(**vars(main_args))
