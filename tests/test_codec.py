import unittest
from preprocess.prepare import parse_melody_to_beats_notes
from utils.render import convert_to_melody
import numpy as np
class Tests(unittest.TestCase):
    MELODY = [(1.2, 3.6, 60), (4.8, 7.9, 61), (7.9, 8.2, 62), (8.2, 8.3, 63)]
    BEATS = np.array([
            [1.2, 3.6-1.2],
            [4.8-3.6, 7.9-4.8],
            [7.9-7.9, 8.2-7.9],
            [8.2-8.2, 8.3-8.2],
            ])

    def test_parse_melody_to_beats_notes(self):
        beats, notes = parse_melody_to_beats_notes(self.MELODY)
        
        beats_expected = self.BEATS
        notes_expected = np.array([60, 61, 62, 63]).reshape(-1, 1)

        self.assertTrue(np.allclose(beats, beats_expected))
        self.assertTrue(np.allclose(notes, notes_expected))
    def test_render(self):
        start_time, end_time, note = convert_to_melody(self.BEATS, np.array([x[2] for x in self.MELODY]))
        self.assertEqual(len(start_time), len(end_time))
        self.assertEqual(len(start_time), len(note))
        self.assertEqual(len(start_time), len(self.MELODY))
        for i in range(len(self.MELODY)):
            self.assertAlmostEqual(start_time[i], self.MELODY[i][0])
            self.assertAlmostEqual(end_time[i], self.MELODY[i][1])
            self.assertAlmostEqual(note[i], self.MELODY[i][2])


if __name__ == '__main__':
    unittest.main()