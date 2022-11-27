from preprocess.dataset import BeatsRhythmsDataset
import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("index", type=int, default="index")
    args.add_argument("-o", "--output", type=str, default="output.mid")
    args = args.parse_args()
    dataset = BeatsRhythmsDataset(64)
    dataset.load()
    # idx = dataset.name_to_idx["2011/MIDI-Unprocessed_22_R1_2011_MID--AUDIO_R1-D8_12_Track12_wav.midi"]
    idx = args.index
    dataset.to_midi(idx, args.output)
    print(dataset.notes_list[idx][:256].reshape(-1))