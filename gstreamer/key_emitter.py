from enum import Enum

class KeyState(Enum):
    RELEASED = 0
    PRESSED = 2
    REPEATING = 3

class RawCode(Enum):
    NO_INPUT   = -1
    GEST_UP    = 0
    GEST_DOWN  = 1
    GEST_LEFT  = 2
    GEST_RIGHT = 3
    GEST_OK    = 4
    GEST_BACK  = 5
    GEST_EXIT  = 6
    GEST_MENU  = 7
    GEST_PAUSE = 8
    GEST_FACE  = 10

class KeyCode(Enum):
    NO_KEY = 0
    UP     = 0x81
    DOWN   = 0x82
    LEFT   = 0x83
    RIGHT  = 0x84
    OK     = 0x85
    BACK   = 0x95
    MENU   = 0xc0
    PAUSE  = 0x9b
    FACE   = 0x1

class KeyEvent(Enum):
    NO_EVENT = 0
    PRESS    = 8000
    RELEASE  = 8100
    REPEAT   = 8200
    PRESS_RELEASE = 8300

KEY_PRESS_FIRE_PERIOD  = 2.000
KEY_REPEAT_FIRE_PERIOD = 0.5000

class KeyEmitter:
    def __init__(self):
        self.state = KeyState.RELEASED
        self.raw_input = RawCode.NO_INPUT
        self.input_time = 0

    def rawinput_2_keycode(self, raw_input):
        if raw_input == RawCode.GEST_UP:
            return KeyCode.UP
        elif raw_input == RawCode.GEST_DOWN:
            return KeyCode.DOWN
        elif raw_input == RawCode.GEST_LEFT:
            return KeyCode.LEFT
        elif raw_input == RawCode.GEST_RIGHT:
            return KeyCode.RIGHT
        elif raw_input == RawCode.GEST_OK:
            return KeyCode.OK
        elif raw_input == RawCode.GEST_BACK:
            return KeyCode.BACK
        elif raw_input == RawCode.GEST_EXIT:
            return KeyCode.BACK # raw Exit and Back are mapped to the same key BACK
        elif raw_input == RawCode.GEST_MENU:
            return KeyCode.MENU
        elif raw_input == RawCode.GEST_PAUSE:
            return KeyCode.PAUSE
        elif raw_input == RawCode.GEST_FACE:
            return KeyCode.FACE
        return KeyCode.NO_KEY

    def push_input(self, raw_input, time):
        raw_input = RawCode(raw_input)
        ret_key_code = KeyCode.NO_KEY
        ret_key_event = KeyEvent.NO_EVENT
        ret_fill_ratio = 0.0

        if self.raw_input != raw_input:
            self.input_time = time
            if self.state != KeyState.RELEASED:
                ret_key_code = self.rawinput_2_keycode(self.raw_input) 
                ret_key_event = KeyEvent.RELEASE
                self.state = KeyState.RELEASED
            self.raw_input = raw_input

        if ret_key_event != KeyEvent.RELEASE and raw_input != RawCode.NO_INPUT:
            threshold_delta = KEY_PRESS_FIRE_PERIOD if self.state in {KeyState.RELEASED, KeyState.PRESSED} else KEY_REPEAT_FIRE_PERIOD
            key_fire_time = self.input_time + threshold_delta 
            if time >= key_fire_time:
                self.input_time = time
                ret_key_code = self.rawinput_2_keycode(raw_input)
                ret_fill_ratio = 1.0

                if self.state == KeyState.RELEASED:
                    ret_key_event = KeyEvent.PRESS
                    self.state = KeyState.PRESSED
                elif self.state == KeyState.PRESSED:
                    ret_key_event = KeyEvent.REPEAT
                    self.state = KeyState.REPEATING
                elif self.state == KeyState.REPEATING:
                    ret_key_event = KeyEvent.REPEAT
            else:
                time_elapsed = time - self.input_time
                if time_elapsed > 0:
                    ret_fill_ratio = time_elapsed / threshold_delta

        return ret_key_code, ret_key_event, ret_fill_ratio

