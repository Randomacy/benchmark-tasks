"""On the station table: a tan board figure of Tom seated across the table, facing the robot, so his right is on the
robot's left. A grey tray lies in front of him. A black keyboard lies on the table to the robot's left of the tray,
its long side running away from the robot, its number pad at the far end. The keyboard is loose.

Run it:  inspect-robots run --embodiment yam_arena -E task=als_keyboard.py --scorer task_score \\
           --instruction "Tom has ALS and can no longer speak, but he can still move his right hand a little. Set up his keyboard on his tray so he can type." ...
"""

import math

import numpy as np

from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import Waypoint, pick_place_waypoints
from yam_arena.task_file import TABLE_TOP, Asset, Box, Cylinder

SCENE = "station"
TOP = TABLE_TOP[SCENE]

TAN = (0.80, 0.66, 0.52)
BOARD = 0.01  # the figure's thickness (m)
TRAY_H = 0.002
TRAY_CENTRE, TRAY_SIZE = (0.30, 0.09), (0.16, 0.26)
HAND_XY = (0.40, 0.12)  # his working (right) hand, on the robot's left

FIGURE = [
    Box("shoulders", pos=(0.47, 0.0, TOP + BOARD / 2), size=(0.06, 0.26, BOARD), color=(0.45, 0.60, 0.50)),
    Box("right_forearm", pos=(0.42, 0.12, TOP + BOARD / 2), size=(0.04, 0.04, BOARD), color=TAN),
    Box("left_forearm", pos=(0.42, -0.12, TOP + BOARD / 2), size=(0.04, 0.04, BOARD), color=TAN),
    Cylinder("head", pos=(0.47, 0.0, TOP + BOARD + 0.025), radius=0.05, height=0.05, color=TAN),
]
TRAY = Box("tray", pos=(*TRAY_CENTRE, TOP + TRAY_H / 2), size=(*TRAY_SIZE, TRAY_H), color=(0.62, 0.62, 0.66))

# Measured in the simulator: at extent 0.26 the keyboard is about 26 x 7.9 x 2.2 cm, its long side along x at yaw 0, with the
# number pad at +x. Typed from its space-bar edge, it faces -y at yaw 0, so it faces Tom (+x) once turned by +90 degrees.
KEYBOARD_XY = (0.22, 0.27)
KEYBOARD = Asset("keyboard", asset="keyboard_hot3d_robolab", xy=KEYBOARD_XY, extent=0.26, mass=0.2, convex=True)
OBJECTS = [*FIGURE, TRAY, KEYBOARD]

FACING_YAW_DEG = 90.0  # the turn from its starting pose that puts the space-bar edge toward Tom
YAW_TOLERANCE_DEG = 15.0
FLAT_TILT_DEG = 10.0
REACH = 0.14  # its centre must be this close to his working hand (m)

_START = {}  # the keyboard's rotation on the first step of a trial (a library model's axes may not match the world's)


def _rot(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _note_start(objects):
    if "keyboard" not in _START:
        _START["keyboard"] = _rot(objects["keyboard"].quat_xyzw)
        _START["low_z"] = float(objects["keyboard"].low[2])


def _turn(state):
    """The keyboard's rotation since the start, as (yaw about the vertical, tilt from flat), in degrees."""
    rel = _rot(state.quat_xyzw) @ _START["keyboard"].T
    yaw = math.degrees(math.atan2(rel[1, 0], rel[0, 0]))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(rel[2, 2])))))
    return yaw, tilt


def _yaw_off(yaw, target):
    """Smallest angle between two yaws (degrees)."""
    return abs((yaw - target + 180.0) % 360.0 - 180.0)


def _on_tray(state):
    over = (
        abs(state.pos[0] - TRAY_CENTRE[0]) < TRAY_SIZE[0] / 2 and abs(state.pos[1] - TRAY_CENTRE[1]) < TRAY_SIZE[1] / 2
    )
    # Measured from its own start: the model's bounding box reaches 7 mm below the table it rests on.
    resting_on_tray = abs(float(state.low[2]) - _START["low_z"] - TRAY_H) < 0.005
    return bool(over and resting_on_tray and _turn(state)[1] < FLAT_TILT_DEG and state.resting())


def _facing(state):
    """Its space-bar edge is toward him, so the keys read the right way up from his side."""
    return _yaw_off(_turn(state)[0], FACING_YAW_DEG) < YAW_TOLERANCE_DEG


def _in_reach(state):
    return math.hypot(state.pos[0] - HAND_XY[0], state.pos[1] - HAND_XY[1]) <= REACH


def success(objects):
    _note_start(objects)
    k = objects["keyboard"]
    return bool(_on_tray(k) and _facing(k) and _in_reach(k))


def distance(objects):
    return float(np.hypot(*(objects["keyboard"].pos[:2] - np.array(TRAY_CENTRE))))


def score(objects):
    """Partial credit (success scores 1.0): 0.25 on the tray, and if it is on the tray, 0.5 for facing him and 0.25 for
    being within reach of his working hand. A keyboard placed backwards is no use to him, so it can reach only 0.5."""
    _note_start(objects)
    k = objects["keyboard"]
    if not _on_tray(k):
        return 0.0
    return 0.25 + 0.5 * _facing(k) + 0.25 * _in_reach(k)


GRIP_H = 0.014  # the fingertips reach 1.3 cm below the grip point: this keeps them 1 mm above the table
TRAVEL_Z = TOP + 0.14
LEFT_PARK = (0.20, 0.30, TRAVEL_Z)


def demo():
    """The left arm grips the keyboard across its narrow side, turns it a quarter turn while carrying it, and lays it on
    the tray with the space-bar edge toward Tom."""
    steps = pick_place_waypoints((*KEYBOARD_XY, TOP + GRIP_H), (*TRAY_CENTRE, TOP + TRAY_H + GRIP_H + 0.005),
                                 travel_z=TRAVEL_Z, yaw=0.0, place_yaw=math.radians(FACING_YAW_DEG), speed=0.7)
    steps.append(Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "clear"))
    return {"left": steps}
