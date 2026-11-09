"""On the station table: a dinner plate and a curved plastic plate guard standing on the table beside it. The guard
clips onto the plate's rim: its upper jaw is spring-loaded and rests 3 mm lower than the rim's top, with a bevelled
nose. Pressed sideways onto the rim, the guard must force that jaw up, and the push shoves the light plate across the
table unless the plate is braced on its far side. So one arm must steady the plate while the other presses the guard
on. The plate and the guard are loose.

Run it:  inspect-robots run --embodiment yam_arena -E task=plate_guard.py --scorer task_score \\
           --instruction "Raj has Parkinson's and keeps pushing his food off the edge of his plate. Clip the plate guard onto his plate for him." ...
"""

import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from shapekit import Compound, Sprung, gripper_points, lift  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

WHITE = (0.94, 0.94, 0.92)
BLUE = (0.45, 0.70, 0.90)

# The plate: a foot 16 cm across and 1.2 cm tall, and a flat rim 22 cm across and 6 mm thick on top of it.
FOOT_R, FOOT_H = 0.08, 0.012
RIM_R, RIM_T = 0.11, 0.006
PLATE_XY = (0.30, 0.02)
PLATE = Compound(
    "plate", pos=(*PLATE_XY, TOP + FOOT_H / 2), radius=FOOT_R, height=FOOT_H, color=WHITE, mass=0.25,
    friction=(0.5, 0.4),
    shapes=(("cyl", (0, 0, FOOT_H / 2 + RIM_T / 2), RIM_R, RIM_T, WHITE),),
)

# The guard, in a frame whose origin is the middle of its back wall at table level. Its arc is centred RB along +y and
# spans 70 degrees in five straight pieces. The back wall and the lower jaw are one rigid body. The lower jaw is a
# block standing on the table whose top is 1 mm below the rim. The upper jaw is a separate piece on a vertical slide
# whose spring holds it 3 mm lower than the rim's top. A 45-degree bevel on its leading edge lets the rim lift it as the
# guard slides on, and the spring then clamps the rim.
WALL_T, WALL_H = 0.004, 0.050
LOW_TOP = FOOT_H - 0.001  # the lower jaw's top
UP_Z, UP_T = FOOT_H + RIM_T - 0.003, 0.004  # the upper jaw's underside at rest, and its thickness
BEVEL = 0.004  # half the diagonal of the square bar that forms the upper jaw's bevelled nose
CLIP_RATE, CLIP_TRAVEL = 400.0, 0.010  # N/m (1.2 N at the 3 mm the rim lifts it), and how far it can lift
RB = RIM_R + 0.002 + WALL_T / 2  # back wall centre from the arc centre: 2 mm clear of the rim's edge
JAW_IN = RIM_R - 0.012  # the jaws reach 12 mm in over the rim
ARC_DEG, PIECES = 70.0, 5
_body, _jaw = [], []
for _i in range(PIECES):
    _a = math.radians(-ARC_DEG / 2 + ARC_DEG * (_i + 0.5) / PIECES)
    _step = math.radians(ARC_DEG / PIECES)

    def _piece(r, depth, z0, z1, a=_a, tilt=None):
        chord = 2 * r * math.sin(_step / 2) + 0.002
        centre = (r * math.sin(a), RB - r * math.cos(a), (z0 + z1) / 2)
        shape = ("box", centre, (chord, depth, z1 - z0), BLUE, math.degrees(a))
        return shape if tilt is None else shape + (tilt,)

    _inner = RB - WALL_T / 2
    _jr, _jd = (_inner + JAW_IN) / 2, _inner - JAW_IN
    _side = BEVEL * math.sqrt(2)
    _body += [_piece(RB, WALL_T, 0.0, WALL_H), _piece(_jr, _jd, 0.0, LOW_TOP)]
    _jaw += [
        _piece(_jr, _jd, UP_Z, UP_Z + UP_T),
        # the nose: a square bar turned 45 degrees about its length, its lower vertex on the jaw's underside
        ("box", (JAW_IN * math.sin(_a), RB - JAW_IN * math.cos(_a), UP_Z + BEVEL),
         (2 * JAW_IN * math.sin(_step / 2) + 0.002, _side, _side), BLUE, math.degrees(_a), 45.0),
    ]
GUARD_START = (0.30, -0.22)  # the origin, on the table
GUARD = Sprung(
    "guard", pos=(GUARD_START[0], GUARD_START[1], TOP), radius=0.0015, height=WALL_H, color=BLUE, mass=0.03,
    friction=(0.35, 0.3), shapes=tuple(_body), moving=tuple(_jaw), travel=CLIP_TRAVEL, rate=CLIP_RATE,
)
OBJECTS = [PLATE, GUARD]

SEATED_M = 0.009  # the rim at least this far into the slot
PARTLY_M = 0.004
SIDEWAYS_M = 0.015  # the guard's middle within this of the plate's centre line along the arc
UPRIGHT_DEG = 12.0
PLATE_FLAT_DEG = 6.0
PLATE_TIPPED_DEG = 20.0
PLATE_MOVED_M = 0.012  # a plate of food shoved further than this across the table, at any point, has been disturbed
HANDS_CLEAR_M = 0.10
STILL_M = 0.002
STILL_CALLS = 60  # scoring calls without moving (at most three a step, so at least 1 s)
_anchors = {}
_trial = {"shoved": False}  # set once the plate has been shoved too far; sliding it back afterwards does not undo it


def _rot(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _tilt_deg(item):
    up = float((_rot(item.quat_xyzw) @ np.array([0.0, 0.0, 1.0]))[2])
    return math.degrees(math.acos(max(-1.0, min(1.0, up))))


def _still(name, item):
    """Moved under 2 mm over about 1 s, from the position: a body the physics engine has put to sleep keeps reporting its
    last velocity, so ``resting()`` can stay false for an object that has stopped."""
    pos = np.array(item.pos, dtype=float)
    anchor = _anchors.get(name)
    if anchor is None or np.linalg.norm(pos - anchor[0]) > STILL_M:
        _anchors[name] = [pos, 0]
        return False
    anchor[1] += 1
    return anchor[1] >= STILL_CALLS


def _guard_origin(guard):
    """The guard's frame origin (its state reports the body origin, not a bounding-box centre) and rotation."""
    return np.array(guard.pos, dtype=float), _rot(guard.quat_xyzw)


def overlap(objects):
    """How far the plate's rim reaches into the guard's slot (m), measured in the guard's frame; with the guard's middle
    off to one side along the arc, the guard not upright or the rim not between its jaws, it is 0."""
    guard, plate = objects["guard"], objects["plate"]
    origin, R = _guard_origin(guard)
    if _tilt_deg(guard) > UPRIGHT_DEG:
        return 0.0
    slot_low = float(origin[2]) + LOW_TOP - 0.002
    slot_high = float(origin[2]) + UP_Z + lift("guard") + 0.002  # the upper jaw's underside, wherever the rim has lifted it
    rim_low = float(plate.low[2]) + FOOT_H
    if not (slot_low <= rim_low and rim_low + RIM_T <= slot_high):  # the rim must lie between the jaws
        return 0.0
    p = R.T @ (np.array(plate.pos, dtype=float) - (origin + R @ np.array([0.0, RB, 0.0])))
    if abs(float(p[0])) > SIDEWAYS_M:
        return 0.0
    return max(0.0, 0.012 - float(p[1]))


def _hands_clear(objects):
    points = gripper_points()
    if points is None:
        return True
    for name in ("plate", "guard"):
        low, high = np.array(objects[name].low, dtype=float), np.array(objects[name].high, dtype=float)
        if min(np.linalg.norm(p - np.clip(p, low, high)) for p in points) < HANDS_CLEAR_M:
            return False
    return True


def _plate_ok(objects):
    plate, guard = objects["plate"], objects["guard"]
    moved = float(np.hypot(*(np.array(plate.pos[:2], dtype=float) - np.array(PLATE_XY))))
    guard_moved = float(np.hypot(*(np.array(guard.pos[:2], dtype=float) - np.array(GUARD_START))))
    if moved < 0.002 and guard_moved < 0.002:  # both where a trial starts: a new trial
        _trial["shoved"] = False
    if moved >= PLATE_MOVED_M:
        _trial["shoved"] = True
    return bool(_tilt_deg(plate) < PLATE_TIPPED_DEG and abs(float(plate.low[2]) - TOP) < 0.02 and not _trial["shoved"])


def success(objects):
    plate, guard = objects["plate"], objects["guard"]
    return bool(_plate_ok(objects) and overlap(objects) >= SEATED_M and _tilt_deg(plate) < PLATE_FLAT_DEG and abs(float(plate.low[2]) - TOP) < 0.005
                and _still("plate", plate) and _still("guard", guard) and _hands_clear(objects))


def distance(objects):
    origin, _ = _guard_origin(objects["guard"])
    return float(np.linalg.norm(origin[:2] - (np.array(objects["plate"].pos[:2]) - np.array([0.0, RB]))))


def score(objects):
    """0 if the plate is tipped past 20 degrees or off the table, or has at any point been shoved 1.2 cm or more from
    where it stood. Otherwise 0.5 once the rim is 4 mm into the guard's slot, 0.75 at 9 mm, and 1.0 at 9 mm with the
    plate flat, both still and both grippers clear."""
    if not _plate_ok(objects):
        return 0.0
    if success(objects):
        return 1.0
    depth = overlap(objects)
    return 0.75 if depth >= SEATED_M else 0.5 if depth >= PARTLY_M else 0.0


TRAVEL_Z = TOP + 0.16
GRIP_UP = 0.044  # grip the back wall this far above the table, with the fingertips clear of the upper jaw
HOME_Y = PLATE_XY[1] - RB  # where the guard's origin sits when it is fully on
PUSH_Y = HOME_Y + 0.035  # aim past the seat: the arm yields like a soft spring, so a far aim gives a firm press
BRACE = (PLATE_XY[0], PLATE_XY[1] + RIM_R + 0.003, TOP + 0.018)  # closed fingers aimed 6 mm into the far edge
LEFT_PARK = (0.20, 0.30, TRAVEL_Z)
RIGHT_PARK = (0.20, -0.30, TRAVEL_Z)


def _slide_on(wait):
    gx, gz = GUARD_START[0], TOP + GRIP_UP
    return [
        Waypoint(wait, RIGHT_PARK, FINGER_OPEN, "wait"),
        Waypoint(2.5, (gx, GUARD_START[1], TRAVEL_Z), FINGER_OPEN, "above the guard"),
        Waypoint(1.5, (gx, GUARD_START[1], gz), FINGER_OPEN, "descend"),
        Waypoint(0.8, (gx, GUARD_START[1], gz), FINGER_CLOSED, "grip its back wall"),
        Waypoint(3.0, (gx, HOME_Y - 0.03, gz), FINGER_CLOSED, "slide it up to the rim"),
        Waypoint(4.0, (gx, PUSH_Y, gz), FINGER_CLOSED, "press it on, the rim lifting the sprung jaw"),
        Waypoint(0.8, (gx, PUSH_Y, gz), FINGER_OPEN, "let go"),
        Waypoint(1.5, (gx, PUSH_Y, TRAVEL_Z), FINGER_OPEN, "rise"),
        Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "clear"),
    ]


def demo():
    """The left arm braces the plate's far edge with closed fingers. The right arm lifts the guard, lines its slot up
    with the rim and pushes it on, lets go and clears. Then the left arm lets go of the plate and clears."""
    if os.environ.get("GUARD_WRONG") == "no_brace":  # scripted check: nothing steadies the plate
        right = _slide_on(1.0)
        return {"left": [Waypoint(sum(w.duration for w in right), LEFT_PARK, FINGER_OPEN, "wait")], "right": right}
    brace = [
        Waypoint(2.5, (BRACE[0], BRACE[1] + 0.03, TRAVEL_Z), FINGER_CLOSED, "above the plate's far edge"),
        Waypoint(1.5, (BRACE[0], BRACE[1] + 0.03, BRACE[2]), FINGER_CLOSED, "descend beside it"),
        Waypoint(1.5, BRACE, FINGER_CLOSED, "brace the edge"),
    ]
    t_braced = sum(w.duration for w in brace)
    right = _slide_on(t_braced + 0.5)
    left = brace + [
        Waypoint(sum(w.duration for w in right) - t_braced, BRACE, FINGER_CLOSED, "hold"),
        Waypoint(1.0, (BRACE[0], BRACE[1] + 0.03, BRACE[2]), FINGER_CLOSED, "back off"),
        Waypoint(1.5, (BRACE[0], BRACE[1] + 0.03, TRAVEL_Z), FINGER_CLOSED, "rise"),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "clear"),
    ]
    return {"left": left, "right": right}
