"""On the station table: a folding reading stand lying flat, its back toward the robot and its front toward the reader on
the far side. The stand's board is hinged along its front edge. A stay hinged under the board swings down as the board
rises, and its foot rides along a ratchet in the base: three notches hold the board at about 30, 45 and 60 degrees.
Raised only part way and let go, the board drops back into a lower notch. Released past the top notch, it settles in
the steepest setting. Everything is loose.

Run it (from the folder holding this file and shapekit.py):
    inspect-robots run --embodiment yam_arena --policy yam_scripted \\
      -E task=$PWD/reading_stand.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \\
      --instruction "Ann has severe arthritis and can't hold a book or look down to read. Set her reading stand up at its steepest setting." \\
      --scorer task_score --max-steps 2400 --store-frames --log-dir runs/demo

Set RS_LOG=1 to write the board's angle and the stay's foot position to /tmp/reading_stand.txt five times a second.
"""

import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from shapekit import Sprung, gripper_points, joints, piece_point  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

GREY = (0.45, 0.47, 0.50)
WOOD = (0.80, 0.65, 0.45)
DARK = (0.20, 0.20, 0.22)
RED = (0.80, 0.20, 0.15)

# --- The stand, in a frame whose origin is the middle of its base on the table. The reader sits on +x. -----------------
# Base: 31 x 32 cm, 1 cm thick, with pads at the sides that the folded board rests on (1.5 cm up). The board: 25 x 30 cm,
# 8 mm thick, hinged along its front edge (x +12 cm), with a book ledge along that edge and a pull tab at the back.
STAND_XY = (0.30, -0.02)
BASE_TOP = 0.010
BOARD_LOW, BOARD_T = 0.025, 0.008
HINGE = (0.12, 0.0, BOARD_LOW + BOARD_T / 2)
BOARD_BACK = -0.13
TAB = (-0.1175, 0.0, BOARD_LOW + BOARD_T + 0.015)  # the pull tab's middle: 1.5 x 2.5 x 3 cm
# The stay: 14 cm long, hinged under the board 10 cm behind the hinge. At rest it lies folded back under the board.
STAY_PIVOT = (0.02, 0.0, BOARD_LOW - 0.003)
STAY_L, FOOT_R = 0.14, 0.005
FOOT = (STAY_PIVOT[0] - STAY_L, 0.0, STAY_PIVOT[2])
# Ratchet ridges 6 mm tall across the middle 20 cm of the base; the gaps between them are the notches.
RIDGE_TOP = BASE_TOP + 0.006
NOTCHES = {30: (-0.102, -0.090), 45: (-0.076, -0.064), 60: (-0.044, -0.032)}  # x ranges, by nominal angle
_ridges = [(-0.090, -0.076), (-0.064, -0.044), (-0.032, 0.000)]

_base = [
    ("box", (-0.025, 0.0, BASE_TOP / 2), (0.31, 0.32, BASE_TOP), GREY),
    ("box", (0.0, 0.135, (BASE_TOP + BOARD_LOW) / 2), (0.24, 0.02, BOARD_LOW - BASE_TOP), GREY),
    ("box", (0.0, -0.135, (BASE_TOP + BOARD_LOW) / 2), (0.24, 0.02, BOARD_LOW - BASE_TOP), GREY),
] + [("box", ((a + b) / 2, 0.0, (BASE_TOP + RIDGE_TOP) / 2), (b - a, 0.20, RIDGE_TOP - BASE_TOP), DARK)
     for a, b in _ridges]
_board = [
    ("box", ((BOARD_BACK + HINGE[0]) / 2, 0.0, BOARD_LOW + BOARD_T / 2), (HINGE[0] - BOARD_BACK, 0.30, BOARD_T), WOOD),
    ("box", (HINGE[0] - 0.0075, 0.0, BOARD_LOW + BOARD_T + 0.0075), (0.015, 0.30, 0.015), WOOD),
    ("box", TAB, (0.015, 0.025, 0.03), RED),
]
_stay = [
    ("box", ((STAY_PIVOT[0] + FOOT[0]) / 2, 0.0, STAY_PIVOT[2]), (STAY_L, 0.012, 0.006), DARK),
    ("box", FOOT, (0.20, 1.4 * FOOT_R, 1.4 * FOOT_R), DARK, 90.0, 45.0),  # a rod across, diamond in section
]
STAND = Sprung(
    "stand", pos=(*STAND_XY, TOP), radius=0.002, height=0.01, color=GREY, mass=0.8, friction=(0.8, 0.7),
    shapes=tuple(_base), moving=tuple(_board), moving_mass=0.45,
    hinge="Y", pivot=HINGE, lower=0.0, upper=80.0, rate=0.0, damping=0.02,
    # The stay can only lean back under the raised board: at 60 degrees it stays at least 5 degrees behind vertical.
    more=((tuple(_stay), 0.06, "Y", STAY_PIVOT, -145.0, 0.0, 0.0),),
    self_collide=True,  # the stay's foot must meet the ratchet in the base
)
OBJECTS = [STAND]

TARGET = 60
ANGLE_TOL_DEG = 5.0
HANDS_CLEAR_M = 0.05
STILL_DEG = 0.5
STILL_CALLS = 60
_anchor = {"a": None, "n": 0}
_log = {"file": None, "calls": 0}


def board_deg():
    j = joints("stand")
    return math.degrees(j[0]) if j else 0.0


def _foot():
    return piece_point("stand", 1, FOOT)  # body frame


def notch():
    """The notch the stay's foot sits in (by nominal angle), or None."""
    f = _foot()
    if f[2] > RIDGE_TOP:
        return None
    for deg, (a, b) in NOTCHES.items():
        if a - 0.002 <= f[0] <= b + 0.002:
            return deg
    return None


def _still():
    a = board_deg()
    if _anchor["a"] is None or abs(a - _anchor["a"]) > STILL_DEG:
        _anchor.update(a=a, n=0)
        return False
    _anchor["n"] += 1
    return _anchor["n"] >= STILL_CALLS


def _settled_in():
    """The notch the board rests in: the foot in a notch and the board within 5 degrees of that notch's angle."""
    n = notch()
    if n is None or abs(board_deg() - n) > ANGLE_TOL_DEG:
        return None
    return n


def _hands_clear(objects):
    points = gripper_points()
    if points is None:
        return True
    low, high = np.array(objects["stand"].low, dtype=float), np.array(objects["stand"].high, dtype=float)
    return min(np.linalg.norm(p - np.clip(p, low, high)) for p in points) >= HANDS_CLEAR_M


def _upright(objects):
    x, y, z, w = objects["stand"].quat_xyzw
    return 1 - 2 * (x * x + y * y) > math.cos(math.radians(10))


def _write_log(objects, s):
    if not os.environ.get("RS_LOG"):
        return
    _log["calls"] += 1
    if _log["calls"] % 4:
        return
    if _log["file"] is None:
        _log["file"] = open("/tmp/reading_stand.txt", "w")
    f = _foot()
    j = joints("stand")
    _log["file"].write(f"{_log['calls']} board={board_deg():.1f}deg stay={math.degrees(j[1]) if len(j) > 1 else 0:.1f}deg "
                       f"foot=({100 * f[0]:.1f},{100 * f[2]:.1f})cm notch={notch()} score={s}\n")
    _log["file"].flush()


def success(objects):
    return bool(_upright(objects) and _settled_in() == TARGET and _still() and _hands_clear(objects))


def distance(objects):
    return abs(board_deg() - TARGET) / 100.0


def score(objects):
    """0 unless the board rests on its stay in a notch. 0.25 in the 30 or 45 degree notch; 0.75 in the steepest
    (60 degree) notch; 1.0 there with the board still and both grippers clear."""
    if success(objects):
        s = 1.0
    elif not _upright(objects):
        s = 0.0
    else:
        n = _settled_in()
        s = 0.75 if n == TARGET else 0.25 if n is not None else 0.0
    _write_log(objects, s)
    return s


# --- Scripted demo ------------------------------------------------------------------------------------------------
GRIP_UP = 0.01  # grip the tab this far above its middle
LEFT_PARK = (0.20, 0.30, TOP + 0.16)
RIGHT_PARK = (0.20, -0.30, TOP + 0.25)


def tab_at(deg):
    """Where to hold the pull tab (station frame) with the board raised to ``deg``."""
    rel = np.array(TAB) + np.array([0.0, 0.0, GRIP_UP]) - np.array(HINGE)
    a = math.radians(deg)
    x = rel[0] * math.cos(a) + rel[2] * math.sin(a)
    z = -rel[0] * math.sin(a) + rel[2] * math.cos(a)
    return (STAND_XY[0] + HINGE[0] + x, STAND_XY[1] + rel[1], TOP + HINGE[2] + z)


def demo():
    """The right arm grips the board's pull tab and raises the board along its arc to 70 degrees, past the top notch,
    lowers it to 58 degrees so the stay's foot drops into that notch, lets go and clears."""
    over = tab_at(0)
    if os.environ.get("RS_WRONG") == "low":  # scripted check: let go at 50 degrees, short of the top notch
        return _raise_and_let_go(over, (12, 24, 36, 50), 50)
    return _raise_and_let_go(over, (12, 24, 36, 48, 60, 70), 58)


def _raise_and_let_go(over, angles, let_go_at):
    right = [
        Waypoint(1.0, RIGHT_PARK, FINGER_OPEN, "wait"),
        Waypoint(2.5, (over[0], over[1], TOP + 0.15), FINGER_OPEN, "above the tab"),
        Waypoint(2.0, over, FINGER_OPEN, "descend"),
        Waypoint(1.0, over, FINGER_CLOSED, "grip the tab"),
    ]
    for deg in angles:
        right.append(Waypoint(1.2, tab_at(deg), FINGER_CLOSED, f"raise to {deg} degrees"))
    top = angles[-1]
    right += [
        Waypoint(1.5, tab_at(top), FINGER_CLOSED, "hold"),
        Waypoint(2.0, tab_at(let_go_at), FINGER_CLOSED, "lower onto the stay"),
        Waypoint(1.0, tab_at(let_go_at), FINGER_OPEN, "let go"),
        Waypoint(1.5, (tab_at(let_go_at)[0] - 0.08, tab_at(let_go_at)[1], tab_at(let_go_at)[2] + 0.06), FINGER_OPEN,
                 "back away"),
        Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "clear"),
    ]
    return {"left": [Waypoint(sum(w.duration for w in right), LEFT_PARK, FINGER_OPEN, "wait")], "right": right}
