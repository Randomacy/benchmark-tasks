# Get a non-slip mat under a heavy mixing bowl for a person with Parkinson's

A heavy mixing bowl of batter stands on a worktop with a non-slip mat beside it. Asked to stop the bowl moving for someone
with Parkinson's, the robot has to work out that the mat goes under the bowl. The bowl cannot be carried level by its
handle: lifting the handle tips it about the far edge of its foot. So one arm must hold it tipped, without spilling, while
the other gets the mat in underneath, and then the bowl must be lowered level onto the mat.

Files: `task.json`, `parkinsons_mat.py` (scene, `success()`, `score()`, scripted demo), `shapekit.py` (lets one rigid body
have several shapes, for the bowl and its handle, and reads the gripper positions for the release check), `README.md`.
Keep `shapekit.py` next to the task file. No third-party assets are used: the bowl, its handle and the mat are built from
boxes and cylinders in `parkinsons_mat.py`.

## Changes in this revision

In response to the review:

- **No displacement veto.** The 4 cm limit on moving the bowl is gone. Moving the bowl is allowed: lifting or sliding it
  aside and setting it level on the mat, without spilling, can earn full credit. The only veto left is spilling (the bowl
  tipped past 30 degrees at any moment).
- **Partial credit needs real overlap.** 0.25 now requires the mat to overlap the outline of the bowl's foot by at least
  2 cm, measured as the foot's radius less the distance from the foot's centre to the nearest point of the mat, in the
  mat's own frame. A mat beside or beyond the bowl scores nothing.
- **Release check.** Success now also requires every finger link of both grippers to be at least 5 cm from the bowl's and
  the mat's outlines (`shapekit.gripper_points()`).
- The foot's position follows the bowl when it is moved or turned (the offset from the reported bounding-box centre is
  kept in the bowl's own frame).

## Environment

Built from the program's Brev launchable (Isaac Lab-Arena, L40S). Versions used for the submitted runs:

- inspect-robots 0.60.0
- yam-arena at commit 0059d02a1a91
- task file SHA-256, first 12 hex: 2e36c397eb52 (`sha256sum parkinsons_mat.py | cut -c1-12`; equals `eval.environment_revision` in the logs)
- `shapekit.py` SHA-256, first 12 hex: ab99e698fc19

## Run a trial

On the launchable, from the folder holding `parkinsons_mat.py`:

```bash
INSTRUCTION="Joan has Parkinson's, and her mixing bowl keeps moving on the worktop while she stirs. Sort this out for her with the non-slip mat."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$PWD/parkinsons_mat.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTRUCTION" --scorer task_score \
  --max-steps 2400 --store-frames --log-dir runs/dream

# Scripted full-score demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$PWD/parkinsons_mat.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "$INSTRUCTION" --scorer task_score \
  --max-steps 2400 --store-frames --log-dir runs/demo
```

`--scorer task_score` reports the score: 1.0 on success, otherwise the partial credit from `score()` at the end of the
episode. A trial ends once success has held for 1 s.

## Wrong strategies, run in the simulator

The demo takes `BOWL_WRONG` from the environment. Add it in front of the demo command to run these:

| Run | What it does | Score | Log |
|---|---|---|---|
| `BOWL_WRONG=carry` | One arm lifts the bowl clear by its handle, and it hangs steeply: the batter spills | 0.0 | `adhoc_81ad6d07.json` |
| `BOWL_WRONG=no_lift` | Pushes the mat at the bowl without tipping it; the mat stops at the foot, overlapping it by under 2 cm | 0.0 | `adhoc_435382db.json` |

## Runtime and seeds

- Run limit: 2,400 control steps (2 minutes at 20 Hz).
- Fixed layout: `-E object_jitter=0`, seed 0 (the default) for every run; no other seed is needed.

## Submitted runs

- Demo: log `adhoc_1885a18b.json`, `task_score = 1.0`, ended by `success` after 664 frames.
- Dream-1 run: log `adhoc_bfc03406.json`, `task_score = 0.0`, ended by `max_steps` after 2,400 steps.
- Setup screenshot: frame 0 of the top camera from the demo.
- Videos: `inspect-robots video <log>`, then the `top_cam` stream re-encoded to H.264.

## Notes

- Stillness is measured from position (under 2 mm over about 1 s), not from `resting()`: a body the physics engine has put
  to sleep against another can keep reporting its last velocity, so `resting()` can stay false for something that has stopped.
- The reported centre of the bowl is its bounding box's, which the handle pulls toward the robot. The scorer corrects for
  this with the offset seen at the start of the trial, kept in the bowl's own frame so that it follows the bowl.
