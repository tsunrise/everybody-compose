from pynput import keyboard
import numpy as np
import time

from utils.data_paths import DataPaths
from preprocess.prepare import convert_start_end_to_beats
from enum import Enum

class Event(Enum):
    PRESS = 1
    RELEASE = 0

def create_beat():
    '''
    Parse user's key presses into a sequence of beats.
    Record user's space key pressing time until the enter key is pressed
    Return a numpy 2d array in shape of (seq_length, 2) representing beat sequence that user enters.
    each row of the array is a beat represented by [prev_rest_time, duration]
    '''

    TAP_KEYS = {keyboard.KeyCode.from_char('z'), keyboard.KeyCode.from_char('x'), keyboard.Key.space}
    ENTER_KEY = keyboard.Key.enter

    events = []
    pressing_key = None
    base_time = time.time()
    def on_press(key):
        '''
        listener that monitor presses of the space key
        record the previous rest time until the space key is pressed
        '''
        nonlocal events, pressing_key
        if key in TAP_KEYS and key != pressing_key:
            curr_time = time.time() - base_time
            if pressing_key is not None:
                events.append((Event.RELEASE, curr_time))
            events.append((Event.PRESS, curr_time))
            pressing_key = key

    def on_release(key):
        '''
        listener that monitor release of the space key and enter key
        record the pressed time on the space key
        stop the listener when the enter key is released
        '''
        nonlocal events, pressing_key
        if key == ENTER_KEY:
            # Stop listener
            curr_time = time.time() - base_time
            if pressing_key is not None:
                events.append((Event.RELEASE, curr_time))
            return False
        elif key in TAP_KEYS and key == pressing_key:
            events.append((Event.RELEASE, time.time() - base_time))
            pressing_key = None


    print("use z,x,space key on keyboard to create a sequence of beat")
    print("hit enter to stop")
    with keyboard.Listener(on_press=on_press, on_release=on_release, suppress=True) as listener:
        listener.join()

    # convert events to start_time and end_time
    start_time = []
    end_time = []
    num_pressed = 0
    for event, timestamp in events:
        if event == Event.PRESS:
            start_time.append(timestamp)
            if num_pressed > 0:
                end_time.append(timestamp)
            num_pressed += 1
        else:
            if num_pressed == 1:
                end_time.append(timestamp)
            num_pressed -= 1
    assert len(start_time) == len(end_time)
    assert num_pressed == 0
    # print("start_time: ", start_time)
    # print("end_time: ", end_time)
    beat_sequence = convert_start_end_to_beats(np.array(start_time), np.array(end_time))

    paths = DataPaths()
    file_name = "last_recorded.npy"
    file_path = paths.beats_rhythms_dir / file_name
    np.save(file_path, beat_sequence)

    return beat_sequence