import argparse
from preprocess.prepare import prepare_raw_beats_notes


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Prepare Dataset')
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--mono', type=bool, default=True)
    parser.add_argument('--max_files', type=int, default=-1)
    parser.add_argument('--save-freq', type=int, default=500)
    args = parser.parse_args()
    
    seq_len = args.seq_len
    mono = args.mono
    max_files = None if args.max_files == -1 else args.max_files
    beats, notes = prepare_raw_beats_notes(mono, max_files, progress_save_freq=args.save_freq, override=True)
    print(f"len(beats): {len(beats)}, len(notes): {len(notes)}")
    
    