from pynput import keyboard
import numpy as np
import time

from utils.data_paths import DataPaths

def create_beat():
    '''
    Parse user's key presses into a sequence of beats.
    Record user's space key pressing time until the enter key is pressed
    Return a numpy 2d array in shape of (seq_length, 2) representing beat sequence that user enters.
    each row of the array is a beat represented by [prev_rest_time, duration]
    '''
    prev_rest, duration = [], []
    press_time, release_time = time.time(), time.time()
    pressed = False

    def on_press(key):
        '''
        listener that monitor presses of the space key
        record the previous rest time until the space key is pressed
        '''
        nonlocal pressed, press_time
        if key == keyboard.Key.space and pressed == False:
            pressed = True
            press_time = time.time()
            prev_rest.append(round(press_time - release_time, 2))

    def on_release(key):
        '''
        listener that monitor release of the space key and enter key
        record the pressed time on the space key
        stop the listener when the enter key is released
        '''
        nonlocal pressed, release_time
        if key == keyboard.Key.space:
            release_time = time.time()
            duration.append(round(release_time - press_time, 2))
            pressed = False
        if key == keyboard.Key.enter:
            return False


    print("use space key on keyboard to create a sequence of beat")
    print("hit enter to stop")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
    beat_sequence = np.column_stack((prev_rest, duration))
    paths = DataPaths()
    file_name = "last_recorded.npy"
    file_path = paths.beats_rhythms_dir / file_name
    np.save(file_path, beat_sequence)

    return beat_sequence