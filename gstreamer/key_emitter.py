from enum import Enum

class KeyState(Enum):
    RELEASED = 0
    PRESSED = 2
    REPEATING = 3

class RawCode(Enum):
    NO_INPUT = -1
    GEST_UP = 0
    GEST_DOWN = 1
    GEST_OK = 2

class KeyCode(Enum):
    NO_KEY = 0
    UP = 45
    DOWN = 46
    OK = 47

class KeyEvent(Enum):
    NO_EVENT = 0
    PRESSED  = 8001
    REPEAT   = 8002
    RELEASED = 8003

KEY_PRESS_FIRE_PERIOD  = 2000 * 1000
KEY_REPEAT_FIRE_PERIOD = 1000 * 1000

class KeyEmitter:
    def __init__(self):
        self.state = KeyState.RELEASED
        self.raw_input = RawCode.NO_INPUT
        self.input_time = 0

    def rawinput_2_keycode(self, raw_input):
        if raw_input == RawCode.NO_INPUT:
            return KeyCode.NO_KEY
        elif raw_input == RawCode.GEST_UP:
            return KeyCode.UP
        elif raw_input == RawCode.GEST_DOWN:
            return KeyCode.DOWN
        elif raw_input == RawCode.GEST_OK:
            return KeyCode.OK

    def push_input(self, raw_input, time):
        ret_key_code = KeyCode.NO_KEY
        ret_key_event = KeyEvent.NO_EVENT

        if self.raw_input != raw_input:
            self.input_time = time
            if self.state != KeyState.RELEASED:
                ret_key_code = self.rawinput_2_keycode(self.raw_input) 
                ret_key_event = KeyEvent.RELEASED
                self.state = KeyState.RELEASED
            self.raw_input = raw_input

        if ret_key_event != KeyEvent.RELEASED and raw_input != RawCode.NO_INPUT:
            time_delta = time - self.input_time
            threshold_delta = KEY_PRESS_FIRE_PERIOD if self.state in {KeyState.RELEASED, KeyState.PRESSED} else KEY_REPEAT_FIRE_PERIOD

            if time_delta >= threshold_delta:
                self.input_time = time
                ret_key_code = self.rawinput_2_keycode(raw_input)

                if self.state == KeyState.RELEASED:
                    ret_key_event = KeyEvent.PRESSED
                    self.state = KeyState.PRESSED
                elif self.state == KeyState.PRESSED:
                    ret_key_event = KeyEvent.REPEAT
                    self.state = KeyState.REPEATING
                elif self.state == KeyState.REPEATING:
                    ret_key_event = KeyEvent.REPEAT


        return ret_key_code, ret_key_event

