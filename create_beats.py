from utils.beats_generator import create_beat
import numpy as np

if __name__ == "__main__":
    beat_sequence = create_beat()
    np.set_printoptions(precision=4)
    print(beat_sequence)
    # save the beat sequence to a file
    np.save("beat_sequence.npy", beat_sequence)
    print("beat sequence saved to beat_sequence.npy")