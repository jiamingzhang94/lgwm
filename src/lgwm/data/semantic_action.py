"""Canonical action schema shared by data, training, and inference."""

ACTION_TYPES = (
    "tap",
    "long_press",
    "swipe",
    "scroll",
    "type_text",
    "key",
    "open_app",
    "wait",
    "click",
    "hover",
    "drag",
)

ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTION_TYPES)}

DIRECTION_LITERAL = {"up": 0, "down": 1, "left": 2, "right": 3}
