import enum
from gstreamer.key_emitter import KeyEmitter
from gstreamer.key_emitter import RawCode
from gstreamer.key_emitter import KeyCode
from gstreamer.key_emitter import KeyEvent
import unittest

KEY_PRESS_THRESHOLD = 2000 * 1000

TIMESTAMP_START                 = 0000 * 1000
TIMESTAMP_KEY_DOWN              = 1000 * 1000
TIMESTAMP_KEY_RELEASE_NOKEY     = 1001 * 1000
TIMESTAMP_KEY_PRESS             = 3000 * 1000
TIMESTAMP_KEY_RELEASE           = 3001 * 1000
TIMESTAMP_KEY_REPEAT_01         = 5000 * 1000
TIMESTAMP_KEY_REPEAT_02         = 6000 * 1000
TIMESTAMP_KEY_RELEASE_REPEAT_02 = 6001 * 1000


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
                (RawCode.NO_INPUT, TIMESTAMP_START,             (KeyCode.NO_KEY, KeyEvent.NO_EVENT)),
                (RawCode.GEST_UP,  TIMESTAMP_KEY_DOWN,          (KeyCode.NO_KEY, KeyEvent.NO_EVENT)),
                (RawCode.NO_INPUT, TIMESTAMP_KEY_RELEASE_NOKEY, (KeyCode.NO_KEY, KeyEvent.NO_EVENT)),
                ]

        for i, received, expected in self.enumerate_and_execute(key_sequence):
            self.assertEqual(received, expected, "Failed at index: {}".format(i))

    def test_push_input_press_key(self):
        key_sequence = [
                (RawCode.NO_INPUT,  TIMESTAMP_START,       (KeyCode.NO_KEY, KeyEvent.NO_EVENT)),
                (RawCode.GEST_DOWN, TIMESTAMP_KEY_DOWN,    (KeyCode.NO_KEY, KeyEvent.NO_EVENT)),
                (RawCode.GEST_DOWN, TIMESTAMP_KEY_PRESS,   (KeyCode.DOWN,   KeyEvent.PRESSED)),
                (RawCode.NO_INPUT,  TIMESTAMP_KEY_RELEASE, (KeyCode.DOWN,   KeyEvent.RELEASED)),
                (RawCode.GEST_DOWN, TIMESTAMP_KEY_RELEASE + TIMESTAMP_KEY_DOWN,    (KeyCode.NO_KEY, KeyEvent.NO_EVENT)),
                (RawCode.GEST_DOWN, TIMESTAMP_KEY_RELEASE + TIMESTAMP_KEY_PRESS,   (KeyCode.DOWN,   KeyEvent.PRESSED)),
                (RawCode.NO_INPUT,  TIMESTAMP_KEY_RELEASE + TIMESTAMP_KEY_RELEASE, (KeyCode.DOWN,   KeyEvent.RELEASED)),
                ]

        for i, received, expected in self.enumerate_and_execute(key_sequence):
            self.assertEqual(received, expected, "Failed at index: {}".format(i))

    def test_push_input_repeat_key(self):
        key_sequence = [
                (RawCode.NO_INPUT, TIMESTAMP_START,                 (KeyCode.NO_KEY, KeyEvent.NO_EVENT)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_DOWN,              (KeyCode.NO_KEY, KeyEvent.NO_EVENT)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_PRESS,             (KeyCode.OK,     KeyEvent.PRESSED)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_REPEAT_01,         (KeyCode.OK,     KeyEvent.REPEAT)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_REPEAT_02,         (KeyCode.OK,     KeyEvent.REPEAT)),
                (RawCode.NO_INPUT, TIMESTAMP_KEY_RELEASE_REPEAT_02, (KeyCode.OK,     KeyEvent.RELEASED)),
                ]

        for i, received, expected in self.enumerate_and_execute(key_sequence):
            self.assertEqual(received, expected, "Failed at index: {}".format(i))

    def test_push_input_repeat_key_check_no_duplicate_event(self):
        key_sequence = [
                (RawCode.NO_INPUT, TIMESTAMP_START,                     (KeyCode.NO_KEY, KeyEvent.NO_EVENT)),
                (RawCode.NO_INPUT, TIMESTAMP_START + 1,                 (KeyCode.NO_KEY, KeyEvent.NO_EVENT)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_DOWN,                  (KeyCode.NO_KEY, KeyEvent.NO_EVENT)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_DOWN + 1,              (KeyCode.NO_KEY, KeyEvent.NO_EVENT)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_PRESS,                 (KeyCode.OK,     KeyEvent.PRESSED)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_PRESS + 1,             (KeyCode.NO_KEY, KeyEvent.NO_EVENT)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_REPEAT_01,             (KeyCode.OK,     KeyEvent.REPEAT)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_REPEAT_01 + 1,         (KeyCode.NO_KEY, KeyEvent.NO_EVENT)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_REPEAT_02,             (KeyCode.OK,     KeyEvent.REPEAT)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_REPEAT_02 + 1,         (KeyCode.NO_KEY, KeyEvent.NO_EVENT)),
                (RawCode.NO_INPUT, TIMESTAMP_KEY_RELEASE_REPEAT_02,     (KeyCode.OK,     KeyEvent.RELEASED)),
                (RawCode.GEST_OK,  TIMESTAMP_KEY_RELEASE_REPEAT_02 + 1, (KeyCode.NO_KEY, KeyEvent.NO_EVENT)),
                ]

        for i, received, expected in self.enumerate_and_execute(key_sequence):
            self.assertEqual(received, expected, "Failed at index: {}".format(i))

    def test_push_input_press_key_on_other_press_key(self):
        key_sequence = [
                (RawCode.NO_INPUT,  TIMESTAMP_START,       (KeyCode.NO_KEY, KeyEvent.NO_EVENT)),
                (RawCode.GEST_DOWN, TIMESTAMP_KEY_DOWN,    (KeyCode.NO_KEY, KeyEvent.NO_EVENT)),
                (RawCode.GEST_DOWN, TIMESTAMP_KEY_PRESS,   (KeyCode.DOWN,   KeyEvent.PRESSED)),
                (RawCode.GEST_OK,   TIMESTAMP_KEY_RELEASE, (KeyCode.DOWN,   KeyEvent.RELEASED)),
                (RawCode.GEST_OK,   TIMESTAMP_KEY_RELEASE + 1,                       (KeyCode.NO_KEY, KeyEvent.NO_EVENT)),
                (RawCode.GEST_OK,   TIMESTAMP_KEY_RELEASE + KEY_PRESS_THRESHOLD,     (KeyCode.OK,   KeyEvent.PRESSED)),
                (RawCode.NO_INPUT,  TIMESTAMP_KEY_RELEASE + KEY_PRESS_THRESHOLD + 1, (KeyCode.OK,   KeyEvent.RELEASED)),
                ]

        for i, received, expected in self.enumerate_and_execute(key_sequence):
            self.assertEqual(received, expected, "Failed at index: {}".format(i))


if __name__ == '__main__':
    unittest.main()
