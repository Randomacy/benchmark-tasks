"""On the station table: a fixed three-level wooden rack, a reduced-scale kitchen rack with its levels stepping up away
from the robot. A red cake tin sits on the narrow middle shelf, a green tea tin on the top shelf to the robot's right,
and a mug on the bottom shelf to its left. The tins and the mug are loose.

Run it:  inspect-robots run --embodiment yam_arena -E task=arthritis_storage.py --scorer task_score \\
           --instruction "Mrs Evans has arthritis in her hands and shoulders, so bending down and reaching up both hurt. On this rack the bottom shelf is at her knees, the middle shelf is between her waist and shoulders and the top shelf is above her head. She uses her mug and the green tea tin every day and the red cake tin only at Christmas. Rearrange the rack for her." ...
"""

import math

import numpy as np

from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import Waypoint, pick_place_waypoints
from yam_arena.task_file import TABLE_TOP, Asset, Box, Cylinder

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# Each shelf: centre x, size along x, size along y, height of its surface above the table (m).
BOTTOM_X, BOTTOM_DX, BOTTOM_DY, BOTTOM_H = 0.215, 0.12, 0.50, 0.002
MIDDLE_X, MIDDLE_DX, MIDDLE_DY, MIDDLE_H = 0.33, 0.11, 0.18, 0.045  # narrow: the everyday items take up most of it
TOP_X, TOP_DX, TOP_DY, TOP_H = 0.445, 0.12, 0.50, 0.08
CAKE_R, CAKE_H = 0.04, 0.04
TEA_R, TEA_H = 0.03, 0.05


def _shelf(name, x, dx, dy, h, color):
    return Box(name, pos=(x, 0.0, TOP + h / 2), size=(dx, dy, h), color=color)


BOTTOM = _shelf("bottom_shelf", BOTTOM_X, BOTTOM_DX, BOTTOM_DY, BOTTOM_H, (0.62, 0.45, 0.28))
MIDDLE = _shelf("middle_shelf", MIDDLE_X, MIDDLE_DX, MIDDLE_DY, MIDDLE_H, (0.70, 0.52, 0.33))
TOP_SHELF = _shelf("top_shelf", TOP_X, TOP_DX, TOP_DY, TOP_H, (0.55, 0.39, 0.24))

CAKE_XY = (MIDDLE_X, 0.0)
TEA_XY = (TOP_X, -0.15)
MUG_XY = (BOTTOM_X, 0.15)

CAKE_TIN = Cylinder("cake_tin", pos=(*CAKE_XY, TOP + MIDDLE_H + CAKE_H / 2), radius=CAKE_R, height=CAKE_H,
                    color=(0.75, 0.12, 0.12), mass=0.15)
TEA_TIN = Cylinder("tea_tin", pos=(*TEA_XY, TOP + TOP_H + TEA_H / 2), radius=TEA_R, height=TEA_H,
                   color=(0.12, 0.45, 0.22), mass=0.15)
MUG = Asset("mug", asset="mug_ycb_robolab", xy=MUG_XY, extent=0.085, mass=0.3, convex=True)
OBJECTS = [BOTTOM, MIDDLE, TOP_SHELF, CAKE_TIN, TEA_TIN, MUG]

ITEMS = ("cake_tin", "tea_tin", "mug")
UPRIGHT_DEG = 15.0
ON_SHELF_M = 0.008
_UP_LOCAL = {}  # each item's local axis that pointed up at the start (a library model's up axis may not be +z)


def _rot(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _note_start(objects):
    """Record how each item stands on the first step of a trial, while it is still as placed."""
    for name in ITEMS:
        if name not in _UP_LOCAL:
            _UP_LOCAL[name] = _rot(objects[name].quat_xyzw).T @ np.array([0.0, 0.0, 1.0])


def _on_shelf(name, item, x, dx, dy, h):
    """Centre over the shelf, base on its surface, upright and still: not stacked, not held up, not on its side."""
    over = abs(item.pos[0] - x) < dx / 2 and abs(item.pos[1]) < dy / 2
    level = abs(float(item.low[2]) - (TOP + h)) < ON_SHELF_M
    up_z = float((_rot(item.quat_xyzw) @ _UP_LOCAL[name])[2])
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
    return bool(over and level and tilt < UPRIGHT_DEG and item.resting())


def _on_middle(objects, name):
    _note_start(objects)
    return _on_shelf(name, objects[name], MIDDLE_X, MIDDLE_DX, MIDDLE_DY, MIDDLE_H)


def _cake_put_away(objects):
    _note_start(objects)
    cake = objects["cake_tin"]
    return bool(
        _on_shelf("cake_tin", cake, BOTTOM_X, BOTTOM_DX, BOTTOM_DY, BOTTOM_H)
        or _on_shelf("cake_tin", cake, TOP_X, TOP_DX, TOP_DY, TOP_H)
    )


def success(objects):
    return bool(_on_middle(objects, "mug") and _on_middle(objects, "tea_tin") and _cake_put_away(objects))


def distance(objects):
    return float(sum(np.hypot(*(objects[name].pos[:2] - np.array((MIDDLE_X, 0.0)))) for name in ("mug", "tea_tin")))


def score(objects):
    """Partial credit (success scores 1.0): 0.35 each for the mug and the tea tin on the middle shelf, 0.3 for the cake
    tin moved to the bottom or top shelf."""
    return (
        0.35 * _on_middle(objects, "mug")
        + 0.35 * _on_middle(objects, "tea_tin")
        + 0.3 * _cake_put_away(objects)
    )


TRAVEL_Z = TOP + 0.16  # the highest the top-down wrist goes straight down from; the tea tin's top is 13 cm up
CAKE_GRIP_H = 0.02
TEA_GRIP_H = 0.025
MUG_GRIP_H = 0.03
CLEARANCE = 0.006  # the height an item is let go from
CAKE_TARGET = (BOTTOM_X, -0.12)
TEA_TARGET = (MIDDLE_X, -0.05)
MUG_TARGET = (MIDDLE_X, 0.05)
LEFT_PARK = (0.20, 0.30, TRAVEL_Z)
RIGHT_PARK = (0.20, -0.30, TRAVEL_Z)


def demo():
    """The right arm moves the cake tin down to the bottom shelf, which frees the middle shelf, brings the tea tin down to
    it and parks. The left arm waits to the side, then lifts the mug up from the bottom shelf."""
    right = pick_place_waypoints((*CAKE_XY, TOP + MIDDLE_H + CAKE_GRIP_H),
                                 (*CAKE_TARGET, TOP + BOTTOM_H + CAKE_GRIP_H + CLEARANCE), travel_z=TRAVEL_Z, speed=0.5)
    right += pick_place_waypoints((*TEA_XY, TOP + TOP_H + TEA_GRIP_H),
                                  (*TEA_TARGET, TOP + MIDDLE_H + TEA_GRIP_H + CLEARANCE), travel_z=TRAVEL_Z, speed=0.5)
    right.append(Waypoint(1.5, RIGHT_PARK, FINGER_OPEN, "clear"))
    left = [Waypoint(sum(w.duration for w in right), LEFT_PARK, FINGER_OPEN, "wait for the middle shelf to be free")]
    left += pick_place_waypoints((*MUG_XY, TOP + BOTTOM_H + MUG_GRIP_H),
                                 (*MUG_TARGET, TOP + MIDDLE_H + MUG_GRIP_H + CLEARANCE), travel_z=TRAVEL_Z, speed=0.5)
    left.append(Waypoint(1.5, LEFT_PARK, FINGER_OPEN, "clear"))
    return {"left": left, "right": right}
