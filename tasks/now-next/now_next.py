"""On the station table: a tan board figure of Sam seated across the table, facing the robot. In front of him lie a
green Now square, on his left, and an orange Next square, on his right. A red card sits on Now and a blue card on Next.
A yellow card waits on the table to the side, near the robot's left arm. The cards are loose.

Run it:  inspect-robots run --embodiment yam_arena -E task=now_next.py --scorer task_score \\
           --instruction "Sam is autistic and uses a Now and Next board. The green square is Now and the orange square is Next. He has just finished his snack, the red card. The blue card is play and the yellow card is lunch, and lunch comes after play. Update his board." ...
"""

import math

import numpy as np

from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import Waypoint, pick_place_waypoints
from yam_arena.task_file import TABLE_TOP, Box, Cylinder

SCENE = "station"
TOP = TABLE_TOP[SCENE]

TAN = (0.80, 0.66, 0.52)
BOARD = 0.01  # the figure's thickness (m)
PAD_H = 0.002
SQUARE = 0.12
NOW_XY, NEXT_XY = (0.30, -0.08), (0.30, 0.08)  # Sam faces the robot, so his left, where a board starts, is at -y
CARD_W, CARD_H = 0.07, 0.025  # thick enough for the gripper's fingertips to take hold of from above
CARD_MASS = 0.06
YELLOW_XY = (0.18, 0.24)


def _card(name, xy, z, color):
    return Box(name, pos=(*xy, z + CARD_H / 2), size=(CARD_W, CARD_W, CARD_H), color=color, mass=CARD_MASS)


FIGURE = [
    Box("shoulders", pos=(0.47, 0.0, TOP + BOARD / 2), size=(0.06, 0.26, BOARD), color=(0.45, 0.55, 0.75)),
    Box("right_forearm", pos=(0.415, 0.15, TOP + BOARD / 2), size=(0.05, 0.04, BOARD), color=TAN),
    Box("left_forearm", pos=(0.415, -0.15, TOP + BOARD / 2), size=(0.05, 0.04, BOARD), color=TAN),
    Cylinder("head", pos=(0.47, 0.0, TOP + BOARD + 0.025), radius=0.05, height=0.05, color=TAN),
]
NOW = Box("now_square", pos=(*NOW_XY, TOP + PAD_H / 2), size=(SQUARE, SQUARE, PAD_H), color=(0.25, 0.65, 0.35))
NEXT = Box("next_square", pos=(*NEXT_XY, TOP + PAD_H / 2), size=(SQUARE, SQUARE, PAD_H), color=(0.95, 0.60, 0.15))
RED = _card("red_card", NOW_XY, TOP + PAD_H, (0.80, 0.12, 0.12))
BLUE = _card("blue_card", NEXT_XY, TOP + PAD_H, (0.12, 0.25, 0.80))
YELLOW = _card("yellow_card", YELLOW_XY, TOP, (0.95, 0.85, 0.15))
OBJECTS = [*FIGURE, NOW, NEXT, RED, BLUE, YELLOW]

FLAT_TILT_DEG = 10.0
HEIGHT_TOLERANCE = 0.005


def _tilt_deg(card):
    x, y, _, _ = card.quat_xyzw
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def _over(card, xy):
    half = SQUARE / 2
    return bool(abs(card.pos[0] - xy[0]) < half and abs(card.pos[1] - xy[1]) < half)


def _flat_and_still(card):
    return bool(_tilt_deg(card) < FLAT_TILT_DEG and card.resting())


def _on_square(card, xy):
    """Lying flat and still on that square, not stacked on another card or held above it."""
    return bool(_over(card, xy) and abs(float(card.low[2]) - (TOP + PAD_H)) < HEIGHT_TOLERANCE and _flat_and_still(card))


def _removed(card):
    """Off both squares, lying flat and still on the table: not on Sam, not stacked, not off the table."""
    on_table = abs(float(card.low[2]) - TOP) < HEIGHT_TOLERANCE
    return bool(not _over(card, NOW_XY) and not _over(card, NEXT_XY) and on_table and _flat_and_still(card))


def success(objects):
    return bool(
        _removed(objects["red_card"]) and _on_square(objects["blue_card"], NOW_XY)
        and _on_square(objects["yellow_card"], NEXT_XY)
    )


def distance(objects):
    return float(np.hypot(*(objects["blue_card"].pos[:2] - np.array(NOW_XY))))


def score(objects):
    """Partial credit (success scores 1.0): 0.25 for the finished red card taken off the board, 0.5 for the blue card on
    Now, 0.25 for the yellow card on Next."""
    return (
        0.25 * _removed(objects["red_card"])
        + 0.5 * _on_square(objects["blue_card"], NOW_XY)
        + 0.25 * _on_square(objects["yellow_card"], NEXT_XY)
    )


GRIP_H = 0.014  # the fingertips reach 1.3 cm below the grip point: this keeps them 1 mm above whatever the card lies on
CLEARANCE = 0.005  # the height a card is let go from
TRAVEL_Z = TOP + 0.14
RED_DROP_XY = (0.18, -0.24)
LEFT_PARK = (0.20, 0.30, TRAVEL_Z)
RIGHT_PARK = (0.20, -0.30, TRAVEL_Z)


def demo():
    """The right arm lifts the finished red card off the board and parks. The left arm waits, then moves the blue card
    to Now and brings the yellow card to Next, in that order, since each square must be empty before it is filled."""
    right = pick_place_waypoints((*NOW_XY, TOP + PAD_H + GRIP_H), (*RED_DROP_XY, TOP + GRIP_H + CLEARANCE),
                                 travel_z=TRAVEL_Z, speed=0.7)
    right.append(Waypoint(1.5, RIGHT_PARK, FINGER_OPEN, "clear"))
    left = [Waypoint(sum(w.duration for w in right), LEFT_PARK, FINGER_OPEN, "wait for the red card to come off")]
    left += pick_place_waypoints((*NEXT_XY, TOP + PAD_H + GRIP_H), (*NOW_XY, TOP + PAD_H + GRIP_H + CLEARANCE),
                                 travel_z=TRAVEL_Z, speed=0.7)
    left += pick_place_waypoints((*YELLOW_XY, TOP + GRIP_H), (*NEXT_XY, TOP + PAD_H + GRIP_H + CLEARANCE),
                                 travel_z=TRAVEL_Z, speed=0.7)
    left.append(Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "clear"))
    return {"left": left, "right": right}
