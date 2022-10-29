import enum
from gstreamer.key_emitter import KeyEmitter
from gstreamer.key_emitter import RawCode
from gstreamer.key_emitter import KeyCode
from gstreamer.key_emitter import KeyEvent
import unittest

TIMESTAMP_START               = 2000 * 1000
TIMESTAMP_KEY_DOWN            = 3000 * 1000
TIMESTAMP_KEY_RELEASE_NOKEY   = 3001 * 1000
TIMESTAMP_KEY_PRESS           = 5000 * 1000
TIMESTAMP_KEY_RELEASE         = 5001 * 1000
TIMESTAMP_KEY_REPEAT_01       = 7000 * 1000
TIMESTAMP_KEY_REPEAT_02       = 8000 * 1000


class TestKeyEmitter(unittest.TestCase):

    def setUp(self):
        self.emitter = KeyEmitter()

    def test_push_input_press_key_too_short(self):
        self.emitter.push_input(RawCode.NO_INPUT, TIMESTAMP_START)

        key, event = self.emitter.push_input(RawCode.GEST_UP, TIMESTAMP_KEY_DOWN)
        expected = (KeyCode.NO_KEY, KeyEvent.NO_EVENT)
        self.assertEqual((key, event), expected, "Got: {},{}\nShould be: {}".format(key, event, expected))

        key, event = self.emitter.push_input(RawCode.NO_INPUT, TIMESTAMP_KEY_RELEASE_NOKEY)
        expected = (KeyCode.NO_KEY, KeyEvent.NO_EVENT)
        self.assertEqual((key, event), expected, "Got: {},{}\nShould be: {}".format(key, event, expected))

    def test_push_input_press_key(self):
        self.emitter.push_input(RawCode.NO_INPUT, TIMESTAMP_START)
        self.emitter.push_input(RawCode.GEST_DOWN, TIMESTAMP_KEY_DOWN)

        key, event = self.emitter.push_input(RawCode.GEST_DOWN, TIMESTAMP_KEY_PRESS)
        expected = (KeyCode.DOWN, KeyEvent.PRESSED)
        self.assertEqual((key, event), expected, "Got: {},{}\nShould be: {}".format(key, event, expected))

        key, event = self.emitter.push_input(RawCode.NO_INPUT, TIMESTAMP_KEY_RELEASE)
        expected = (KeyCode.DOWN, KeyEvent.RELEASED)
        self.assertEqual((key, event), expected, "Got: {},{}\nShould be: {}".format(key, event, expected))


if __name__ == '__main__':
    unittest.main()

