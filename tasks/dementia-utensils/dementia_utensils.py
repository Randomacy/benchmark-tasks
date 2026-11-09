"""On the station table: a tan board figure of Mrs Ward seated across the table, facing the robot, with a fixed bowl of
soup on a place mat in front of her. Two forks lie on the mat, one each side of the bowl. A blue spoon with a thick
handle lies on a grey pad near the robot's left arm. The forks and the spoon are loose.

Run it:  inspect-robots run --embodiment yam_arena -E task=dementia_utensils.py --scorer task_score \\
           --instruction "Mrs Ward has dementia and too many utensils confuse her. Her lunch today is soup. Get her place ready." ...
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
PAD_H = 0.002
MAT_CENTRE, MAT_SIZE = (0.30, 0.0), (0.18, 0.34)
PAD_CENTRE, PAD_SIZE = (0.16, 0.24), (0.18, 0.09)
BODY_BOXES = {  # name: (centre x, centre y, size x, size y, colour)
    "shoulders": (0.47, 0.0, 0.06, 0.26, (0.62, 0.50, 0.68)),
    "right_forearm": (0.415, 0.15, 0.05, 0.04, TAN),
    "left_forearm": (0.415, -0.15, 0.05, 0.04, TAN),
}
FIGURE = [Box(name, pos=(x, y, TOP + BOARD / 2), size=(sx, sy, BOARD), color=c)
          for name, (x, y, sx, sy, c) in BODY_BOXES.items()]
FIGURE.append(Cylinder("head", pos=(0.47, 0.0, TOP + BOARD + 0.025), radius=0.05, height=0.05, color=TAN))
MAT = Box("mat", pos=(*MAT_CENTRE, TOP + PAD_H / 2), size=(*MAT_SIZE, PAD_H), color=(0.20, 0.35, 0.70))
PAD = Box("side_pad", pos=(*PAD_CENTRE, TOP + PAD_H / 2), size=(*PAD_SIZE, PAD_H), color=(0.60, 0.60, 0.63))

UTENSIL_LEN = 0.14
BOWL = Asset("bowl", asset="bowl_ycb_robolab", xy=MAT_CENTRE, extent=0.13, color=(0.95, 0.95, 0.93), convex=True)
SOUP = Cylinder("soup", pos=(*MAT_CENTRE, TOP + 0.034), radius=0.05, height=0.004, color=(0.88, 0.45, 0.12))
FORK_A_XY, FORK_B_XY = (0.30, 0.12), (0.30, -0.12)
SPOON_XY = PAD_CENTRE
FORK_A = Asset("dinner_fork", asset="fork_big_vomp_robolab", xy=FORK_A_XY, extent=UTENSIL_LEN, mass=0.05, convex=True)
FORK_B = Asset("small_fork", asset="fork_small_vomp_robolab", xy=FORK_B_XY, extent=UTENSIL_LEN, mass=0.05, convex=True)
SPOON = Asset("spoon", asset="spoon_1_handal_robolab", xy=SPOON_XY, extent=UTENSIL_LEN, mass=0.05, convex=True)
OBJECTS = [*FIGURE, MAT, PAD, BOWL, SOUP, FORK_A, FORK_B, SPOON]

UTENSILS = ("dinner_fork", "small_fork", "spoon")
FLAT_DEG = 20.0
LEVEL_M = 0.008
_UP_LOCAL = {}  # each utensil's local axis that pointed up at the start (a library model's up axis may not be +z)


def _rot(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _note_start(objects):
    """Record how each utensil lies on the first step of a trial, while it is still as placed."""
    for name in UTENSILS:
        if name not in _UP_LOCAL:
            _UP_LOCAL[name] = _rot(objects[name].quat_xyzw).T @ np.array([0.0, 0.0, 1.0])


def _flat(name, item):
    up_z = float((_rot(item.quat_xyzw) @ _UP_LOCAL[name])[2])
    return math.degrees(math.acos(max(-1.0, min(1.0, up_z)))) < FLAT_DEG


def _over_mat(item):
    return bool(abs(item.pos[0] - MAT_CENTRE[0]) < MAT_SIZE[0] / 2 and abs(item.pos[1] - MAT_CENTRE[1]) < MAT_SIZE[1] / 2)


def _spoon_set(objects):
    """Centre over the mat, lying flat and still on it: not in the bowl, on a fork or held up."""
    _note_start(objects)
    spoon = objects["spoon"]
    level = abs(float(spoon.low[2]) - (TOP + PAD_H)) < LEVEL_M
    return bool(_over_mat(spoon) and level and _flat("spoon", spoon) and spoon.resting())


def _fork_removed(objects, name):
    """Centre off the mat, lying flat and still at table height: not on her, not stacked, not off the table."""
    _note_start(objects)
    fork = objects[name]
    level = abs(float(fork.low[2]) - TOP) < LEVEL_M
    return bool(not _over_mat(fork) and level and _flat(name, fork) and fork.resting())


def success(objects):
    return bool(_spoon_set(objects) and _fork_removed(objects, "dinner_fork") and _fork_removed(objects, "small_fork"))


def distance(objects):
    return float(np.hypot(*(objects["spoon"].pos[:2] - np.array(MAT_CENTRE))))


def score(objects):
    """Partial credit (success scores 1.0): 0.5 for the spoon on the mat, 0.25 for each fork off it."""
    return (
        0.5 * _spoon_set(objects)
        + 0.25 * _fork_removed(objects, "dinner_fork")
        + 0.25 * _fork_removed(objects, "small_fork")
    )


TRAVEL_Z = TOP + 0.14
FORK_GRIP_H = 0.014  # the fingertips reach 1.3 cm below the grip point: this keeps them 1 mm above what the fork lies on
SPOON_GRIP_H = 0.016
CLEARANCE = 0.005  # the height a utensil is let go from
FORK_A_DROP = (0.13, 0.10)
FORK_B_DROP = (0.15, -0.22)
SPOON_TARGET = FORK_A_XY
LEFT_PARK = (0.20, 0.30, TRAVEL_Z)
RIGHT_PARK = (0.20, -0.30, TRAVEL_Z)


def demo():
    """The right arm takes the small fork off the mat and parks. The left arm takes the dinner fork off the mat, then
    lays the spoon where that fork was. Neither arm goes near the bowl."""
    right = pick_place_waypoints((*FORK_B_XY, TOP + PAD_H + FORK_GRIP_H), (*FORK_B_DROP, TOP + FORK_GRIP_H + CLEARANCE),
                                 travel_z=TRAVEL_Z, speed=0.5)
    right.append(Waypoint(1.5, RIGHT_PARK, FINGER_OPEN, "clear"))
    left = pick_place_waypoints((*FORK_A_XY, TOP + PAD_H + FORK_GRIP_H), (*FORK_A_DROP, TOP + FORK_GRIP_H + CLEARANCE),
                                travel_z=TRAVEL_Z, speed=0.5)
    left += pick_place_waypoints((*SPOON_XY, TOP + PAD_H + SPOON_GRIP_H),
                                 (*SPOON_TARGET, TOP + PAD_H + SPOON_GRIP_H + CLEARANCE), travel_z=TRAVEL_Z, speed=0.5)
    left.append(Waypoint(1.5, LEFT_PARK, FINGER_OPEN, "clear"))
    return {"left": left, "right": right}
