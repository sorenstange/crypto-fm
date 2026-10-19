# Project rules
- At the start of every session, read PROGRESS.md and plan.md before doing anything.
- Before a session ends, update PROGRESS.md and commit.
- The full plan is in plan.md. Work one milestone at a time; stop and report when a milestone is done.
- We are on DTU HPC (LSF). Never run training, preprocessing or other heavy compute on the login node — write a job script in jobs/ and submit it with bsub.
- Data, checkpoints and W&B files live under DATA_ROOT (= <project>/data, in the 30 GB home quota; no work/scratch filesystem is used). Space is tight: delete files that are no longer needed (caches such as ~/.cache, intermediates, superseded checkpoints) as you go.
- Never commit API keys.