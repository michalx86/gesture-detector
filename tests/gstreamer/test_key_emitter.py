import enum
from gstreamer.key_emitter import KeyEmitter
from gstreamer.key_emitter import RawCode
from gstreamer.key_emitter import KeyCode
from gstreamer.key_emitter import KeyEvent
import unittest

class TestKeyEmitter(unittest.TestCase):

    def setUp(self):
        self.emitter = KeyEmitter()

    def test_push_input_press_key(self):
        self.emitter.push_input(RawCode.NO_INPUT,2000)

        key, event = self.emitter.push_input(RawCode.GEST_UP,2030)
        expected = (KeyCode.UP, KeyEvent.PRESSED)
        self.assertEqual((key, event), expected, "Got: {},{}\nShould be: {}".format(key, event, expected))

        key, event = self.emitter.push_input(RawCode.NO_INPUT,2060)
        expected = (KeyCode.UP, KeyEvent.RELEASED)
        self.assertEqual((key, event), expected, "Got: {},{}\nShould be: {}".format(key, event, expected))


if __name__ == '__main__':
    unittest.main()

