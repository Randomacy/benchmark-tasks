# Update a Now and Next board for an autistic child

A `yam_arena` task file for Isaac Lab Arena: `now_next.py` defines the objects, the scorer
(`success()` and `score()`) and the scripted full-score demo.

## Run a trial

On the YAM launchable (Isaac Lab Arena venv), from the folder holding `now_next.py`:

```bash
INSTRUCTION="Sam is autistic and uses a Now and Next board. The green square is Now and the orange square is Next. He has just finished his snack, the red card. The blue card is play and the yellow card is lunch, and lunch comes after play. Update his board."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$PWD/now_next.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTRUCTION" --scorer task_score \
  --max-steps 2400 --store-frames --log-dir runs/dream

# Scripted full-score demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$PWD/now_next.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "$INSTRUCTION" --scorer task_score \
  --max-steps 2400 --store-frames --log-dir runs/demo
```

`--scorer task_score` reports the score: 1.0 on success, otherwise the partial credit from `score()` at
the end of the episode.

## Runtime and seeds

- Run limit: 2,400 control steps (2 minutes at 20 Hz). The scripted demo finishes in 844 control steps
  (about 42 s of simulated time).
- Fixed layout: `-E object_jitter=0`, so no seed is needed; every trial starts from the same poses.
