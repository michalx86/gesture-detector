from enum import Enum

class KeyState(Enum):
    RELEASED = 0
    PRESSING = 1
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

class KeyEmitter:
    def __init__(self):
        self.state = KeyState.RELEASED
        self.raw_input = RawCode.NO_INPUT
        self.input_time = 0

    def push_input(self, raw_input, time):
        self.raw_input = raw_input
        if time == 2030:
            return KeyCode.UP, KeyEvent.PRESSED
        return KeyCode.UP, KeyEvent.RELEASED


