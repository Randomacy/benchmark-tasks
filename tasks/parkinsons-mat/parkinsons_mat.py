"""On the station table, as a kitchen worktop: a heavy mixing bowl of batter with a grip handle facing the robot, and a
non-slip mat lying beside it. The bowl is too heavy to lift level by its handle: raising the handle tips it about its far
edge. To get the mat under it, one arm must hold the bowl tipped while the other slides the mat in from the side.

Run it:  inspect-robots run --embodiment yam_arena -E task=parkinsons_mat.py --scorer task_score \\
           --instruction "Joan has Parkinson's, and her mixing bowl keeps moving on the worktop while she stirs. Sort this out for her with the non-slip mat." ...
"""

import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from shapekit import Compound  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

BOWL_XY = (0.31, 0.10)
FOOT_R, FOOT_H = 0.055, 0.02
GLAZE = (0.20, 0.42, 0.62)
BATTER = (0.93, 0.84, 0.62)
HANDLE_Z = 0.072  # above the table
HANDLE_X = -0.13  # from the bowl's centre, toward the robot
_rel = lambda z: z - FOOT_H / 2  # a height above the table, in the base cylinder's frame
BOWL = Compound(
    "bowl", pos=(*BOWL_XY, TOP + FOOT_H / 2), radius=FOOT_R, height=FOOT_H, color=GLAZE, mass=1.2,
    shapes=(
        ("cyl", (0, 0, _rel(0.03)), 0.068, 0.02, GLAZE),
        ("cyl", (0, 0, _rel(0.05)), 0.080, 0.02, GLAZE),
        ("cyl", (0, 0, _rel(0.07)), 0.090, 0.02, GLAZE),
        ("cyl", (0, 0, _rel(0.09)), 0.098, 0.02, GLAZE),
        ("cyl_look", (0, 0, _rel(0.1005)), 0.088, 0.002, BATTER),
        ("box", (HANDLE_X, 0, _rel(HANDLE_Z)), (0.07, 0.05, 0.015), GLAZE),
    ),
)
MAT_H = 0.008
MAT_SIZE = (0.08, 0.24, MAT_H)
MAT_X = BOWL_XY[0] - 0.02  # covers the bowl's foot from its near edge to 2 cm past its centre
MAT_START_Y, MAT_END_Y = -0.13, 0.07
MAT = Box("mat", pos=(MAT_X, MAT_START_Y, TOP + MAT_H / 2), size=MAT_SIZE, color=(0.13, 0.40, 0.36), mass=0.1)
OBJECTS = [BOWL, MAT]

SPILL_DEG = 30.0  # a bowl of batter tipped past this has spilled
MOVED_M = 0.04  # the bowl's foot further than this from where it stood has been dragged or carried off
FLAT_DEG = 6.0
ON_MAT_M = 0.005
STILL_M = 0.002
STILL_CALLS = 60  # scoring calls without moving (at most three a step, so at least 1 s)
_state = {"failed": False, "offset": None}
_anchors = {}


def _tilt_deg(item):
    x, y, _, _ = item.quat_xyzw
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def _still(name, item):
    """Moved under 2 mm over about 1 s, from the position: a body the physics engine has put to sleep against another
    keeps reporting its last velocity, so ``resting()`` can stay false for an object that has stopped."""
    pos = np.array(item.pos, dtype=float)
    anchor = _anchors.get(name)
    if anchor is None or np.linalg.norm(pos - anchor[0]) > STILL_M:
        _anchors[name] = [pos, 0]
        return False
    anchor[1] += 1
    return anchor[1] >= STILL_CALLS


def _foot_xy(bowl):
    """Where the bowl's foot is: the reported centre is the bounding box's, which the handle pulls toward the robot."""
    if _state["offset"] is None:
        _state["offset"] = np.array(BOWL_XY) - np.array(bowl.pos[:2], dtype=float)
    return np.array(bowl.pos[:2], dtype=float) + _state["offset"]


def _note(objects):
    bowl = objects["bowl"]
    if _tilt_deg(bowl) > SPILL_DEG or np.linalg.norm(_foot_xy(bowl) - np.array(BOWL_XY)) > MOVED_M:
        _state["failed"] = True


def _mat_on_table(mat):
    return bool(abs(float(mat.low[2]) - TOP) < ON_MAT_M and _tilt_deg(mat) < 10.0)


def _mat_partly_under(objects):
    """The mat's end has gone under the foot's outline by at least 2 cm, and it lies flat on the table."""
    mat, foot = objects["mat"], _foot_xy(objects["bowl"])
    return bool(_mat_on_table(mat) and float(mat.high[1]) > foot[1] - FOOT_R + 0.02
                and abs(float(mat.pos[0]) - foot[0]) < MAT_SIZE[0] / 2 + FOOT_R)


def _bowl_centre_over_mat(objects):
    mat, foot = objects["mat"], _foot_xy(objects["bowl"])
    inside = abs(foot[0] - float(mat.pos[0])) < MAT_SIZE[0] / 2 and abs(foot[1] - float(mat.pos[1])) < MAT_SIZE[1] / 2
    return bool(_mat_on_table(mat) and inside)


def _bowl_seated(objects):
    """Standing level on the mat: its lowest point at the mat's top, tilted under 6 degrees, and still."""
    bowl = objects["bowl"]
    level = abs(float(bowl.low[2]) - (TOP + MAT_H)) < ON_MAT_M and _tilt_deg(bowl) < FLAT_DEG
    return bool(_still("bowl", bowl) and level)


def success(objects):
    _note(objects)
    return bool(not _state["failed"] and _bowl_centre_over_mat(objects) and _bowl_seated(objects)
                and _still("mat", objects["mat"]))


def distance(objects):
    return float(np.linalg.norm(_foot_xy(objects["bowl"]) - np.array(objects["mat"].pos[:2], dtype=float)))


def score(objects):
    """0 if the bowl was ever tipped past 30 degrees or moved more than 4 cm. Otherwise 0.25 if the mat has gone at least
    2 cm under the foot, 0.5 if the bowl's centre is over the mat, 1.0 if it also stands level on the mat, still."""
    _note(objects)
    if _state["failed"]:
        return 0.0
    if success(objects):
        return 1.0
    if _bowl_centre_over_mat(objects):
        return 0.5
    return 0.25 if _mat_partly_under(objects) else 0.0


TRAVEL_Z = TOP + 0.16
HOLD_X = BOWL_XY[0] + HANDLE_X
GRIP_Z = TOP + HANDLE_Z
RAISE = 0.075
ARC_X = 0.04  # raising the handle about the foot's far edge also carries it away from the robot
PUSH_Z = TOP + 0.015
PUSH_GAP = 0.018
SIDEWAYS = math.pi / 2
LEFT_PARK = (0.20, 0.30, TRAVEL_Z)
RIGHT_PARK = (0.20, -0.30, TRAVEL_Z)


def _wrong_one_arm():
    """Scripted check: BOWL_WRONG=carry tries to lift the bowl clear by its handle with one arm. It should score 0."""
    left = [
        Waypoint(2.5, (HOLD_X, BOWL_XY[1], TRAVEL_Z), FINGER_OPEN, "above the handle"),
        Waypoint(1.5, (HOLD_X, BOWL_XY[1], GRIP_Z), FINGER_OPEN, "descend"),
        Waypoint(1.0, (HOLD_X, BOWL_XY[1], GRIP_Z), FINGER_CLOSED, "grip the handle"),
        Waypoint(4.0, (HOLD_X, BOWL_XY[1], GRIP_Z + 0.16), FINGER_CLOSED, "lift it clear"),
    ]
    return {"left": left, "right": [Waypoint(sum(w.duration for w in left), RIGHT_PARK, FINGER_OPEN, "wait")]}


def _push_mat(wait):
    """Closed fingers push the mat along the worktop from behind, which keeps it flat: held by one end, it pivots in the
    fingers and its far end rises into the bowl."""
    start, end = MAT_START_Y - MAT_SIZE[1] / 2 - PUSH_GAP, MAT_END_Y - MAT_SIZE[1] / 2 - PUSH_GAP
    return [
        Waypoint(wait, RIGHT_PARK, FINGER_OPEN, "wait"),
        Waypoint(2.5, (MAT_X, start, TRAVEL_Z), FINGER_CLOSED, "behind the mat, fingers closed", SIDEWAYS),
        Waypoint(1.5, (MAT_X, start, PUSH_Z), FINGER_CLOSED, "descend", SIDEWAYS),
        Waypoint(6.0, (MAT_X, end, PUSH_Z), FINGER_CLOSED, "push it along", SIDEWAYS),
        Waypoint(1.0, (MAT_X, end - 0.03, PUSH_Z), FINGER_CLOSED, "back off", SIDEWAYS),
        Waypoint(1.5, (MAT_X, end - 0.03, TRAVEL_Z), FINGER_CLOSED, "rise", SIDEWAYS),
        Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "clear"),
    ]


def _wrong_no_lift():
    """Scripted check: BOWL_WRONG=no_lift pushes the mat at the bowl without tipping it. It should score 0."""
    right = _push_mat(1.0)
    return {"left": [Waypoint(sum(w.duration for w in right), LEFT_PARK, FINGER_OPEN, "wait")], "right": right}


def demo():
    """The left arm grips the handle and tips the bowl about its far edge. While it holds, the right arm slides the mat
    in under the foot from the side and withdraws. The left arm then lowers the bowl onto the mat and lets go."""
    wrong = os.environ.get("BOWL_WRONG")
    if wrong == "carry":
        return _wrong_one_arm()
    if wrong == "no_lift":
        return _wrong_no_lift()
    tipped = (HOLD_X + ARC_X, BOWL_XY[1], GRIP_Z + RAISE)
    lift = [
        Waypoint(2.5, (HOLD_X, BOWL_XY[1], TRAVEL_Z), FINGER_OPEN, "above the handle"),
        Waypoint(1.5, (HOLD_X, BOWL_XY[1], GRIP_Z), FINGER_OPEN, "descend"),
        Waypoint(1.0, (HOLD_X, BOWL_XY[1], GRIP_Z), FINGER_CLOSED, "grip the handle"),
        Waypoint(4.0, tipped, FINGER_CLOSED, "tip the bowl up"),
    ]
    t_tipped = sum(w.duration for w in lift)
    slide = _push_mat(t_tipped + 1.0)
    t_slid = sum(w.duration for w in slide)
    left = lift + [
        Waypoint(t_slid - t_tipped, tipped, FINGER_CLOSED, "hold it up"),
        Waypoint(4.0, (HOLD_X, BOWL_XY[1], GRIP_Z + MAT_H), FINGER_CLOSED, "lower it onto the mat"),
        Waypoint(1.0, (HOLD_X, BOWL_XY[1], GRIP_Z + MAT_H), FINGER_OPEN, "let go"),
        Waypoint(1.5, (HOLD_X, BOWL_XY[1], TRAVEL_Z), FINGER_OPEN, "rise"),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "clear"),
    ]
    return {"left": left, "right": slide}
