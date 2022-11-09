import enum
from gstreamer.key_emitter import KeyEmitter
from gstreamer.key_emitter import RawCode
from gstreamer.key_emitter import KeyCode
from gstreamer.key_emitter import KeyEvent
import unittest

KEY_PRESS_THRESHOLD = 2.000

TIMESTAMP_START                 = 0.000
TIMESTAMP_KEY_DOWN              = 1.000
TIMESTAMP_KEY_RELEASE_NOKEY     = 1.001
TIMESTAMP_KEY_PRESS             = 3.000
TIMESTAMP_KEY_RELEASE           = 3.001
TIMESTAMP_KEY_REPEAT_01         = 5.000
TIMESTAMP_KEY_REPEAT_02         = 6.000
TIMESTAMP_KEY_RELEASE_REPEAT_02 = 6.001

EMPTY_FILL_RATIO = 0.0
PRESS_ALMOST_EMPTY_FILL_RATIO = 0.0004999999999999449 # ~ 0.001 / KEY_PRESS_THRESHOLD
REPEAT_ALMOST_EMPTY_FILL_RATIO = 0.001000000000000334 # ~ 0.001 / KEY_REPEAT_THRESHOLD
FULL_FILL_RATIO = 1.0


class TestKeyEmitter(unittest.TestCase):

    def setUp(self):
        self.emitter = KeyEmitter()

    def enumerate_and_execute(self, sequence):
        n = 0
        for raw_code, time, expected in sequence:
            received = self.emitter.push_input(raw_code, time)
            yield n, received, expected
            n += 1

    def test_push_input_press_key_too_short(self):
        key_sequence = [
                (RawCode.NO_INPUT, TIMESTAMP_START,             (KeyCode.NO_KEY, KeyEvent.NO_EVENT, EMPTY_FILL_RATIO)),
                (RawCode.GEST_UP,  TIMESTAMP_KEY_DOWN,          (KeyCode.NO_KEY, KeyEvent.NO_EVENT, EMPTY_FILL_RATIO)),
                (RawCode.NO_INPUT, TIMESTAMP_KEY_RELEASE_NOKEY, (KeyCode.NO_KEY, KeyEvent.NO_EVENT, EMPTY_FILL_RATIO)),
                ]

        for i, received, expected in self.enumerate_and_execute(key_sequence):
            self.assertEqual(received, expected, "Failed at index: {}".format(i))

    def test_push_input_press_key(self):
        key_sequence = [
                (RawCode.NO_INPUT,  TIMESTAMP_START,       (KeyCode.NO_KEY, KeyEvent.NO_EVENT, EMPTY_FILL_RATIO)),
                (RawCode.GEST_DOWN, TIMESTAMP_KEY_DOWN,    (KeyCode.NO_KEY, KeyEvent.NO_EVENT, EMPTY_FILL_RATIO)),
                (RawCode.GEST_DOWN, TIMESTAMP_KEY_PRESS,   (KeyCode.DOWN,   KeyEvent.PRESS,    FULL_FILL_RATIO)),
                (RawCode.NO_INPUT,  TIMESTAMP_KEY_RELEASE, (KeyCode.DOWN,   KeyEvent.RELEASE,  EMPTY_FILL_RATIO)),
                (RawCode.GEST_DOWN, TIMESTAMP_KEY_RELEASE + TIMESTAMP_KEY_DOWN,    (KeyCode.NO_KEY, KeyEvent.NO_EVENT, EMPTY_FILL_RATIO)),
                (RawCode.GEST_DOWN, TIMESTAMP_KEY_RELEASE + TIMESTAMP_KEY_PRESS,   (KeyCode.DOWN,   KeyEvent.PRESS,    FULL_FILL_RATIO)),
                (RawCode.NO_INPUT,  TIMESTAMP_KEY_RELEASE + TIMESTAMP_KEY_RELEASE, (KeyCode.DOWN,   KeyEvent.RELEASE,  EMPTY_FILL_RATIO)),
                ]

        for i, received, expected in self.enumerate_and_execute(key_sequence):
            self.assertEqual(received, expected, "Failed at index: {}".format(i))

    def test_push_input_repeat_key(self):
        key_sequence = [
                (RawCode.NO_INPUT, TIMESTAMP_START,                 (KeyCode.NO_KEY, KeyEvent.NO_EVENT, EMPTY_FILL_RATIO)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_DOWN,              (KeyCode.NO_KEY, KeyEvent.NO_EVENT, EMPTY_FILL_RATIO)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_PRESS,             (KeyCode.OK,     KeyEvent.PRESS,    FULL_FILL_RATIO)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_REPEAT_01,         (KeyCode.OK,     KeyEvent.REPEAT,   FULL_FILL_RATIO)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_REPEAT_02,         (KeyCode.OK,     KeyEvent.REPEAT,   FULL_FILL_RATIO)),
                (RawCode.NO_INPUT, TIMESTAMP_KEY_RELEASE_REPEAT_02, (KeyCode.OK,     KeyEvent.RELEASE,  EMPTY_FILL_RATIO)),
                ]

        for i, received, expected in self.enumerate_and_execute(key_sequence):
            self.assertEqual(received, expected, "Failed at index: {}".format(i))

    def test_push_input_repeat_key_check_no_duplicate_event(self):
        key_sequence = [
                (RawCode.NO_INPUT, TIMESTAMP_START,                         (KeyCode.NO_KEY, KeyEvent.NO_EVENT, EMPTY_FILL_RATIO)),
                (RawCode.NO_INPUT, TIMESTAMP_START + 0.001,                 (KeyCode.NO_KEY, KeyEvent.NO_EVENT, EMPTY_FILL_RATIO)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_DOWN,                      (KeyCode.NO_KEY, KeyEvent.NO_EVENT, EMPTY_FILL_RATIO)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_DOWN + 0.001,              (KeyCode.NO_KEY, KeyEvent.NO_EVENT, PRESS_ALMOST_EMPTY_FILL_RATIO)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_PRESS,                     (KeyCode.OK,     KeyEvent.PRESS,    FULL_FILL_RATIO)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_PRESS + 0.001,             (KeyCode.NO_KEY, KeyEvent.NO_EVENT, PRESS_ALMOST_EMPTY_FILL_RATIO)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_REPEAT_01,                 (KeyCode.OK,     KeyEvent.REPEAT,   FULL_FILL_RATIO)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_REPEAT_01 + 0.001,         (KeyCode.NO_KEY, KeyEvent.NO_EVENT, REPEAT_ALMOST_EMPTY_FILL_RATIO)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_REPEAT_02,                 (KeyCode.OK,     KeyEvent.REPEAT,   FULL_FILL_RATIO)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_REPEAT_02 + 0.001,         (KeyCode.NO_KEY, KeyEvent.NO_EVENT, REPEAT_ALMOST_EMPTY_FILL_RATIO)),
                (RawCode.NO_INPUT, TIMESTAMP_KEY_RELEASE_REPEAT_02,         (KeyCode.OK,     KeyEvent.RELEASE,  EMPTY_FILL_RATIO)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_RELEASE_REPEAT_02 + 0.001, (KeyCode.NO_KEY, KeyEvent.NO_EVENT, EMPTY_FILL_RATIO)),
                ]

        for i, received, expected in self.enumerate_and_execute(key_sequence):
            self.assertEqual(received, expected, "Failed at index: {}".format(i))

    def test_push_input_press_key_on_other_press_key(self):
        key_sequence = [
                (RawCode.NO_INPUT,  TIMESTAMP_START,       (KeyCode.NO_KEY, KeyEvent.NO_EVENT, EMPTY_FILL_RATIO)),
                (RawCode.GEST_DOWN, TIMESTAMP_KEY_DOWN,    (KeyCode.NO_KEY, KeyEvent.NO_EVENT, EMPTY_FILL_RATIO)),
                (RawCode.GEST_DOWN, TIMESTAMP_KEY_PRESS,   (KeyCode.DOWN,   KeyEvent.PRESS,    FULL_FILL_RATIO)),
                (RawCode.GEST_OK,   TIMESTAMP_KEY_RELEASE, (KeyCode.DOWN,   KeyEvent.RELEASE,  EMPTY_FILL_RATIO)),
                (RawCode.GEST_OK,   TIMESTAMP_KEY_RELEASE + 0.001,                       (KeyCode.NO_KEY, KeyEvent.NO_EVENT, PRESS_ALMOST_EMPTY_FILL_RATIO)),
                (RawCode.GEST_OK,   TIMESTAMP_KEY_RELEASE + KEY_PRESS_THRESHOLD,         (KeyCode.OK,     KeyEvent.PRESS,    FULL_FILL_RATIO)),
                (RawCode.NO_INPUT,  TIMESTAMP_KEY_RELEASE + KEY_PRESS_THRESHOLD + 0.001, (KeyCode.OK,     KeyEvent.RELEASE,  EMPTY_FILL_RATIO)),
                ]

        for i, received, expected in self.enumerate_and_execute(key_sequence):
            self.assertEqual(received, expected, "Failed at index: {}".format(i))


if __name__ == '__main__':
    unittest.main()

