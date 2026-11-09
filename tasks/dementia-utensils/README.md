# Set out only the utensil a person with dementia needs for soup

A `yam_arena` task file for Isaac Lab Arena: `dementia_utensils.py` defines the objects, the scorer
(`success()` and `score()`) and the scripted full-score demo.

## Run a trial

On the YAM launchable (Isaac Lab Arena venv), from the folder holding `dementia_utensils.py`:

```bash
INSTRUCTION="Mrs Ward has dementia and too many utensils confuse her. Her lunch today is soup. Get her place ready."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$PWD/dementia_utensils.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTRUCTION" --scorer task_score \
  --max-steps 2400 --store-frames --log-dir runs/dream

# Scripted full-score demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$PWD/dementia_utensils.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "$INSTRUCTION" --scorer task_score \
  --max-steps 2400 --store-frames --log-dir runs/demo
```

`--scorer task_score` reports the score: 1.0 on success, otherwise the partial credit from `score()` at
the end of the episode.

## Runtime and seeds

- Run limit: 2,400 control steps (2 minutes at 20 Hz). The scripted demo finishes in 735 control steps
  (about 37 s of simulated time).
- Fixed layout: `-E object_jitter=0`, so no seed is needed; every trial starts from the same poses.
