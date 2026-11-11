# Set up a folding reading stand at its steepest setting for a person with severe arthritis

A folding reading stand lies flat on the table. Its board is hinged along the edge nearest the reader, and a stay
hinged under the board swings down as the board rises. The stay's foot rides a ratchet in the base: three notches hold
the board at about 30, 45 and 60 degrees. Raised only part way and let go, the board drops back into a lower notch, so
to reach the steepest setting the board has to be lifted past the top notch, lowered until the stay's foot drops in,
and released.

Files: `task.json`, `reading_stand.py` (scene, `success()`, `score()`, scripted demo), `shapekit.py` (rigid parts made
of several shapes, hinged pieces chained into one articulated part, and the gripper positions used by the release
check), `README.md`. Keep `shapekit.py` next to the task file.

## Environment

Built from the program's Brev launchable (Isaac Lab-Arena, L40S). Versions used for the submitted runs:

- inspect-robots 0.60.0
- yam-arena at commit 0059d02a1a91
- task file SHA-256, first 12 hex: fc99cf234d5a (`sha256sum reading_stand.py | cut -c1-12`; equals `eval.environment_revision` in the logs)
- `shapekit.py` SHA-256, first 12 hex: ab99e698fc19

## Run a trial

On the launchable, from the folder holding `reading_stand.py`:

```bash
INSTRUCTION="Ann has severe arthritis and can't hold a book or look down to read. Set her reading stand up at its steepest setting."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$PWD/reading_stand.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTRUCTION" --scorer task_score \
  --max-steps 2400 --store-frames --log-dir runs/dream

# Scripted full-score demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$PWD/reading_stand.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "$INSTRUCTION" --scorer task_score \
  --max-steps 2400 --store-frames --log-dir runs/demo
```

`--scorer task_score` reports the score: 1.0 on success, otherwise the partial credit from `score()` at the end of the
episode. A trial ends once success has held for 1 s.

## The stand

The stand is one articulation of three bodies: the base (the root), the board on a hinge along its far edge, and the
stay on a hinge under the board. Both hinges are free, with light damping. The stay's hinge stops it 145 degrees from
folded, which at the top setting keeps it at least 5 degrees behind vertical, so it props the board from behind. The base and the stay collide with each other, so the stay's foot rides over the ratchet
ridges and drops into the gaps. Measured in the simulator, the top notch holds the board at 58 degrees and the middle
notch at 44 degrees.

## Wrong strategies, run in the simulator

The demo takes `RS_WRONG` from the environment. Add it in front of the demo command:

| Run | What it does | Score | Log |
|---|---|---|---|
| `RS_WRONG=low` | Lifts the board to 50 degrees and lets go there, short of the top notch: the board drops into the middle notch (44 degrees) | 0.25 | `adhoc_8b043c34.json` |

## Runtime and seeds

- Run limit: 2,400 control steps (2 minutes at 20 Hz).
- Fixed layout: `-E object_jitter=0`, seed 0 (the default) for every run.

## Submitted runs

- Demo: log `adhoc_cfe630ff.json`, `task_score = 1.0`, ended by `success` after 404 frames; the board rested at 58
  degrees with the stay's foot in the top notch.
- Dream-1 run: log `adhoc_e03d6b22.json`, `task_score = 0.0`, ended by `max_steps` after 2,400 steps.
- Setup screenshot: frame 0 of the top camera from the demo.
- Videos: `inspect-robots video <log>`, then the `top_cam` stream re-encoded to H.264.

## Notes

- The scorer reads the board's hinge angle and the stay's foot position from the articulation's joints, so it knows
  which notch the foot sits in. A board held up by a gripper with the foot out of a notch scores 0.
- Release check: the scorer reads where each arm's finger links are (`shapekit.gripper_points()`) and requires all of
  them at least 5 cm from the stand's outline.
- Stillness is measured from the board's angle (under 0.5 degrees over about 1 s).
- The demo is open loop: the right arm grips the pull tab, raises the board along its arc to 70 degrees, lowers it to
  58 degrees so the stay's foot drops into the top notch, lets go and clears.
