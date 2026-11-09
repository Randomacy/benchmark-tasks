"""On the station table: a spoon with a thin handle and a hollow foam sleeve, both lying flat. The sleeve's bore leaves
only a few millimetres of play round the handle. One arm must hold the spoon up by its neck, level and still, while the
other carries the sleeve over and threads it along the handle. The spoon is then laid down with the sleeve on it.

Run it:  inspect-robots run --embodiment yam_arena -E task=foam_sleeve.py --scorer task_score \\
           --instruction "Alice has Parkinson's and struggles to hold the thin handle of her spoon. Fit the foam sleeve on it, then lay the spoon down ready for her." ...
"""

import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from shapekit import Compound, gripper_points  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

STEEL = (0.72, 0.74, 0.76)
FOAM = (0.20, 0.22, 0.26)

HEAD_R, HEAD_H = 0.024, 0.012
HANDLE_LEN, HANDLE_W, HANDLE_T = 0.13, 0.012, 0.006
SPOON_LEN = HANDLE_LEN + 2 * HEAD_R
SPOON_XY = (0.30, 0.17)  # the head; the handle points toward the robot
SPOON = Compound(
    "spoon", pos=(*SPOON_XY, TOP + HEAD_H / 2), radius=HEAD_R, height=HEAD_H, color=STEEL, mass=0.04,
    friction=(1.0, 0.8),
    shapes=(("box", (-(HEAD_R + HANDLE_LEN / 2), 0, -(HEAD_H - HANDLE_T) / 2), (HANDLE_LEN, HANDLE_W, HANDLE_T), STEEL),),
)

SLEEVE_LEN, SLEEVE_OD, WALL = 0.09, 0.034, 0.006  # square section; the bore is 22 mm across
SLEEVE_XY = (0.16, -0.14)
_mid = SLEEVE_OD / 2 - WALL / 2  # from the sleeve's axis to the middle of a wall
SLEEVE = Compound(
    "sleeve", pos=(*SLEEVE_XY, TOP + WALL / 2), radius=0.005, height=WALL, color=FOAM, mass=0.02, friction=(0.8, 0.7),
    shapes=(  # in the frame of the base cylinder, which sits inside the bottom wall
        ("box", (0, 0, 0), (SLEEVE_LEN, SLEEVE_OD, WALL), FOAM),
        ("box", (0, 0, 2 * _mid), (SLEEVE_LEN, SLEEVE_OD, WALL), FOAM),
        ("box", (0, _mid, _mid), (SLEEVE_LEN, WALL, SLEEVE_OD - 2 * WALL), FOAM),
        ("box", (0, -_mid, _mid), (SLEEVE_LEN, WALL, SLEEVE_OD - 2 * WALL), FOAM),
    ),
)
OBJECTS = [SPOON, SLEEVE]

ON_AXIS_M = 0.012  # sleeve centre within this of the handle's axis: it can only be there if the handle is in the bore
ON_TABLE_M = 0.008
SPOON_LAID_DEG = 25.0  # with the sleeve under its handle a laid-down spoon leans a little
HANDS_CLEAR_M = 0.10  # gripper links to the outline: measured at most 0.08 m while gripping
STILL_M = 0.002
STILL_CALLS = 60  # scoring calls without moving (at most three a step, so at least 1 s)
_anchors = {}


def _rot(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


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


def covered_length(objects):
    """How much of the handle lies inside the sleeve (m), 0 if the sleeve's centre is off the handle's axis."""
    spoon, sleeve = objects["spoon"], objects["sleeve"]
    axis = _rot(spoon.quat_xyzw) @ np.array([1.0, 0.0, 0.0])  # from the handle's end toward the head
    offset = np.array(sleeve.pos, dtype=float) - np.array(spoon.pos, dtype=float)
    along = float(offset @ axis)
    if np.linalg.norm(offset - along * axis) > ON_AXIS_M:
        return 0.0
    handle_end, handle_top = -SPOON_LEN / 2, -SPOON_LEN / 2 + HANDLE_LEN
    return max(0.0, min(along + SLEEVE_LEN / 2, handle_top) - max(along - SLEEVE_LEN / 2, handle_end))


def _hands_clear(objects):
    points = gripper_points()
    if points is None:
        return True
    for name in ("spoon", "sleeve"):  # measured to the object's outline: a long object's centre can be far from a grip
        low, high = np.array(objects[name].low, dtype=float), np.array(objects[name].high, dtype=float)
        if min(np.linalg.norm(p - np.clip(p, low, high)) for p in points) < HANDS_CLEAR_M:
            return False
    return True


def _laid_down(objects):
    spoon, sleeve = objects["spoon"], objects["sleeve"]
    up = float((_rot(spoon.quat_xyzw) @ np.array([0.0, 0.0, 1.0]))[2])
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, up))))
    on_table = abs(float(spoon.low[2]) - TOP) < ON_TABLE_M and abs(float(sleeve.low[2]) - TOP) < ON_TABLE_M
    return bool(on_table and tilt < SPOON_LAID_DEG and _still("spoon", spoon) and _still("sleeve", sleeve))


def success(objects):
    return bool(covered_length(objects) >= 0.08 and _laid_down(objects) and _hands_clear(objects))


def distance(objects):
    return float(np.linalg.norm(np.array(objects["sleeve"].pos) - np.array(objects["spoon"].pos)))


def score(objects):
    """0.25 once 1 cm of the handle is inside the sleeve, 0.5 at 5 cm, 0.75 at 8 cm, and 1.0 at 8 cm with the spoon and
    sleeve laid on the table, still, and both grippers clear of them."""
    covered = covered_length(objects)
    if success(objects):
        return 1.0
    return 0.75 if covered >= 0.08 else 0.5 if covered >= 0.05 else 0.25 if covered >= 0.01 else 0.0


TRAVEL_Z = TOP + 0.14
GRIP_Z = TOP + 0.014  # the fingertips reach 1.3 cm below the grip point: this keeps them 1 mm above the table
NECK_BACK = HEAD_R + 0.012  # the spoon is held by its neck, near its balance point, so its handle does not droop
NECK_XY = (SPOON_XY[0] - NECK_BACK, SPOON_XY[1])
HOLD_XY = (0.37, 0.06)  # on the left arm's side: reaching across, its wrist runs out of yaw
HOLD_Z = TOP + 0.11
HANDLE_DROP = GRIP_Z - (TOP + HANDLE_T / 2)  # how far the handle's axis lies below the grip point
SLEEVE_GRIP_BACK = 0.02  # grip behind the sleeve's middle, to leave room between the hands
AXIS_Z = HOLD_Z - HANDLE_DROP
AXIS_Y = HOLD_XY[1] - 0.012  # the right arm, reaching across, runs about 1 cm to its left of where it is sent
TAIL_X = HOLD_XY[0] - (HANDLE_LEN - (NECK_BACK - HEAD_R))
END_Y = AXIS_Y + 0.005  # the held spoon sits about 2.6 degrees off the x axis
START_X = TAIL_X - 0.015 - SLEEVE_LEN / 2 - SLEEVE_GRIP_BACK
END_X = HOLD_XY[0] - 0.025 - SLEEVE_LEN / 2 - SLEEVE_GRIP_BACK
LEFT_PARK = (0.20, 0.30, TRAVEL_Z)
RIGHT_PARK = (0.20, -0.30, TRAVEL_Z)


def _wrong_no_hold():
    """Scripted check: SLEEVE_WRONG=no_hold leaves the spoon flat on the table and pushes the sleeve along the table at
    its handle. It should score 0."""
    grip_x, z = SLEEVE_XY[0] - SLEEVE_GRIP_BACK, TOP + SLEEVE_OD / 2
    right = [
        Waypoint(2.5, (grip_x, SLEEVE_XY[1], TRAVEL_Z), FINGER_OPEN, "above the sleeve"),
        Waypoint(1.5, (grip_x, SLEEVE_XY[1], z), FINGER_OPEN, "descend"),
        Waypoint(0.8, (grip_x, SLEEVE_XY[1], z), FINGER_CLOSED, "grip it"),
        Waypoint(3.0, (grip_x - 0.02, SPOON_XY[1], z + 0.002), FINGER_CLOSED, "line it up with the handle"),
        Waypoint(4.0, (grip_x + 0.03, SPOON_XY[1], z + 0.002), FINGER_CLOSED, "push it at the handle"),
        Waypoint(0.8, (grip_x + 0.03, SPOON_XY[1], z + 0.002), FINGER_OPEN, "let go"),
        Waypoint(1.5, (grip_x + 0.03, SPOON_XY[1], TRAVEL_Z), FINGER_OPEN, "rise"),
    ]
    return {"left": [Waypoint(sum(w.duration for w in right), LEFT_PARK, FINGER_OPEN, "wait")], "right": right}


def demo():
    """The left arm lifts the spoon by its neck and holds it level. The right arm lifts the sleeve, lines its bore up
    with the handle, threads it along and lets go. The left arm then lays the spoon on the table and lets go."""
    if os.environ.get("SLEEVE_WRONG") == "no_hold":
        return _wrong_no_hold()
    hold = (*HOLD_XY, HOLD_Z)
    lift = [
        Waypoint(2.5, (*NECK_XY, TRAVEL_Z), FINGER_OPEN, "above the spoon's neck"),
        Waypoint(1.5, (*NECK_XY, GRIP_Z), FINGER_OPEN, "descend"),
        Waypoint(0.8, (*NECK_XY, GRIP_Z), FINGER_CLOSED, "grip the neck"),
        Waypoint(2.0, (*NECK_XY, HOLD_Z), FINGER_CLOSED, "lift"),
        Waypoint(3.0, hold, FINGER_CLOSED, "carry it to the middle"),
    ]
    t_up = sum(w.duration for w in lift)
    grip_x = SLEEVE_XY[0] - SLEEVE_GRIP_BACK
    thread = [
        Waypoint(t_up + 0.5, RIGHT_PARK, FINGER_OPEN, "wait for the spoon"),
        Waypoint(2.5, (grip_x, SLEEVE_XY[1], TRAVEL_Z), FINGER_OPEN, "above the sleeve"),
        Waypoint(1.5, (grip_x, SLEEVE_XY[1], TOP + SLEEVE_OD / 2), FINGER_OPEN, "descend"),
        Waypoint(0.8, (grip_x, SLEEVE_XY[1], TOP + SLEEVE_OD / 2), FINGER_CLOSED, "grip the sleeve"),
        Waypoint(2.0, (grip_x, SLEEVE_XY[1], AXIS_Z), FINGER_CLOSED, "lift to the handle's height"),
        Waypoint(3.0, (START_X, AXIS_Y, AXIS_Z), FINGER_CLOSED, "line it up behind the handle"),
        Waypoint(6.0, (END_X, END_Y, AXIS_Z), FINGER_CLOSED, "thread it along"),
        Waypoint(0.8, (END_X, END_Y, AXIS_Z), FINGER_OPEN, "let go"),
        Waypoint(1.5, (END_X, END_Y, TRAVEL_Z), FINGER_OPEN, "rise"),
        Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "clear"),
    ]
    t_done = sum(w.duration for w in thread)
    if os.environ.get("SLEEVE_WRONG") == "never_release":  # scripted check: it should stop short of 1.0
        down = (*HOLD_XY, GRIP_Z + 0.002)
        return {"left": lift + [Waypoint(t_done - t_up, hold, FINGER_CLOSED, "hold it still"),
                                Waypoint(4.0, down, FINGER_CLOSED, "lay it down"),
                                Waypoint(6.0, down, FINGER_CLOSED, "keep hold")], "right": thread}
    left = lift + [
        Waypoint(t_done - t_up, hold, FINGER_CLOSED, "hold it still"),
        Waypoint(4.0, (*HOLD_XY, GRIP_Z + 0.002), FINGER_CLOSED, "lay it down"),
        Waypoint(0.8, (*HOLD_XY, GRIP_Z + 0.002), FINGER_OPEN, "let go"),
        Waypoint(1.5, (*HOLD_XY, TRAVEL_Z), FINGER_OPEN, "rise"),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "clear"),
    ]
    return {"left": left, "right": thread}
