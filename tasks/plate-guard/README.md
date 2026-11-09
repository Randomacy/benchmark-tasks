# Clip a plate guard onto a plate for a person with Parkinson's

A plate and a curved plastic plate guard stand on the table. The guard clips onto the plate's rim: its upper jaw is
spring-loaded and rests 3 mm lower than the rim's top, with a bevelled front edge. Pressed sideways onto the rim, the
guard has to force that jaw up, and the push shoves the light plate across the table unless something holds it. So one
arm braces the plate's far edge while the other presses the guard on. Then both grippers let go.

Files: `task.json`, `plate_guard.py` (scene, `success()`, `score()`, scripted demo), `shapekit.py` (rigid bodies made of
several shapes, the guard's spring-loaded jaw, and the gripper positions used by the release check), `README.md`. Keep
`shapekit.py` next to the task file.

## Environment

Built from the program's Brev launchable (Isaac Lab-Arena, L40S). Versions used for the submitted runs:

- inspect-robots 0.60.0
- yam-arena at commit 0059d02a1a91
- task file SHA-256, first 12 hex: 9a01bf957585 (`sha256sum plate_guard.py | cut -c1-12`; equals `eval.environment_revision` in the logs)
- `shapekit.py` SHA-256, first 12 hex: d064961f67c4

## Run a trial

On the launchable, from the folder holding `plate_guard.py`:

```bash
INSTRUCTION="Raj has Parkinson's and keeps pushing his food off the edge of his plate. Clip the plate guard onto his plate for him."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$PWD/plate_guard.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTRUCTION" --scorer task_score \
  --max-steps 2400 --store-frames --log-dir runs/dream

# Scripted full-score demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$PWD/plate_guard.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "$INSTRUCTION" --scorer task_score \
  --max-steps 2400 --store-frames --log-dir runs/demo
```

`--scorer task_score` reports the score: 1.0 on success, otherwise the partial credit from `score()` at the end of the
episode. A trial ends once success has held for 1 s.

## The spring clip

The guard is a two-body articulation. The back wall and the lower jaw are one rigid body, 30 g. The upper jaw, 5 g, sits
on a vertical prismatic joint with a spring drive of 400 N/m and up to 1 cm of travel. At rest its underside is 1.5 cm
above the table and the rim's top is at 1.8 cm, so the rim must lift it 3 mm, which takes 1.2 N of spring force. A
45-degree bevel, 4 mm high, along the front of its underside turns the sideways push into that lift. Once the guard is
released, the spring clamps the rim between the jaws. The lower jaw's top is 1 mm below the rim, so the guard rises
that 1 mm and hangs on the rim.

The plate weighs 250 g and slides when pushed. Forcing the bevel over the rim pushes harder than the plate's friction,
so an unbraced plate is shoved along ahead of the guard.

## Wrong strategies, run in the simulator

The demo takes `GUARD_WRONG` from the environment. Add it in front of the demo command:

| Run | What it does | Score | Log |
|---|---|---|---|
| `GUARD_WRONG=no_brace` | The same push with the right arm, while the left arm stays parked. The guard goes on, but the plate is shoved about 2 cm | 0.0 | `adhoc_486ef4fe.json` |

## Runtime and seeds

- Run limit: 2,400 control steps (2 minutes at 20 Hz).
- Fixed layout: `-E object_jitter=0`, seed 0 (the default) for every run.

## Submitted runs

- Demo: log `adhoc_09bae355.json`, `task_score = 1.0`, ended by `success` after 491 frames. The plate moved 6 mm while
  braced, and the rim sat 14 mm into the guard.
- Dream-1 run: log `adhoc_060d705f.json`, `task_score = 0.0`, ended by `max_steps` after 2,400 steps.
- Setup screenshot: frame 0 of the top camera from the demo.
- Videos: `inspect-robots video <log>`, then the `top_cam` stream re-encoded to H.264.

## Notes

- Depth is measured in the guard's frame: the guard's state is its body origin (the middle of the back wall at table
  level), read from the articulation's root pose, and the upper jaw's lift is read from its joint. The rim must lie
  between the lower jaw's top and the upper jaw's underside, each with 2 mm of tolerance.
- A shove of 1.2 cm or more zeroes the trial for good. Sliding the plate back afterwards does not undo it, so one arm
  cannot press the guard on and then return the plate.
- Release check: the scorer reads where each arm's finger links are (`shapekit.gripper_points()`) and requires all of
  them at least 10 cm from the outlines of the plate and the guard.
- Stillness is measured from position (under 2 mm over about 1 s), not from `resting()`, which can stay false for a body
  the physics engine has put to sleep.
- The demo is open loop. The scripted arms give way like soft springs under load, so the right arm aims 3.5 cm past the
  seated position (the back wall stops the guard at the rim), and the left arm aims its closed fingers 6 mm into the
  plate's far edge so it presses back.
