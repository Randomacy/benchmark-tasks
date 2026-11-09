# Fit a foam grip sleeve onto a spoon for a person with Parkinson's

A thin-handled spoon and a foam grip sleeve lie on the table. To fit the sleeve, one arm must hold the spoon up, level
and still, while the other lines the sleeve's bore up with the handle and threads it along: pushed on the table, the
spoon slides away, and the bore leaves only a few millimetres of play. The spoon is then laid down and both grippers let
go.

Files: `task.json`, `foam_sleeve.py` (scene, `success()`, `score()`, scripted demo), `shapekit.py` (one rigid body made
of several shapes, and the gripper positions used by the release check), `README.md`. Keep `shapekit.py` next to the task
file.

## Environment

Built from the program's Brev launchable (Isaac Lab-Arena, L40S). Versions used for the submitted runs:

- inspect-robots 0.60.0
- yam-arena at commit 0059d02a1a91
- task file SHA-256, first 12 hex: e02711424eb6 (`sha256sum foam_sleeve.py | cut -c1-12`; equals `eval.environment_revision` in the logs)
- `shapekit.py` SHA-256, first 12 hex: ae09fd147870

## Run a trial

On the launchable, from the folder holding `foam_sleeve.py`:

```bash
INSTRUCTION="Alice has Parkinson's and struggles to hold the thin handle of her spoon. Fit the foam sleeve on it, then lay the spoon down ready for her."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$PWD/foam_sleeve.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTRUCTION" --scorer task_score \
  --max-steps 2400 --store-frames --log-dir runs/dream

# Scripted full-score demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$PWD/foam_sleeve.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "$INSTRUCTION" --scorer task_score \
  --max-steps 2400 --store-frames --log-dir runs/demo
```

`--scorer task_score` reports the score: 1.0 on success, otherwise the partial credit from `score()` at the end of the
episode. A trial ends once success has held for 1 s.

## Wrong strategies, run in the simulator

The demo takes `SLEEVE_WRONG` from the environment. Add it in front of the demo command:

| Run | What it does | Score | Log |
|---|---|---|---|
| `SLEEVE_WRONG=no_hold` | Leaves the spoon flat and pushes the sleeve along the table at its handle | 0.0 | `adhoc_3aef6693.json` |
| `SLEEVE_WRONG=never_release` | Fits the sleeve and lays the spoon down but never lets go | 0.75 | `adhoc_9000e24b.json` |

## Runtime and seeds

- Run limit: 2,400 control steps (2 minutes at 20 Hz).
- Fixed layout: `-E object_jitter=0`, seed 0 (the default) for every run.

## Submitted runs

- Demo: log `adhoc_3cda4b7e.json`, `task_score = 1.0`, ended by `success` after 770 frames.
- Dream-1 run: log `adhoc_79947f6a.json`, `task_score = 0.0`, ended by `max_steps` after 2,400 steps.
- Setup screenshot: frame 0 of the top camera from the demo.
- Videos: `inspect-robots video <log>`, then the `top_cam` stream re-encoded to H.264.

## Notes

- Release check: the scorer reads where each arm's finger links are (`shapekit.gripper_points()`) and requires all of
  them at least 10 cm from the outlines of the spoon and the sleeve. In the demo they stayed within 8 cm while holding
  and were 13 cm or more away once released.
- Stillness is measured from position (under 2 mm over about 1 s), not from `resting()`, which can stay false for a body
  the physics engine has put to sleep.
- The demo is open loop. It aims about 1 cm to the left of the held handle, because the right arm drifts when reaching
  across, and follows the held spoon's slight yaw.
