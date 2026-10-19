# Project rules
- The full plan is in plan.md. Work one milestone at a time; stop and report when a milestone is done.
- We are on DTU HPC (LSF). Never run training, preprocessing or other heavy compute on the login node — write a job script in jobs/ and submit it with bsub.
- Data, checkpoints and W&B files live under DATA_ROOT on the work filesystem, never in home.
- Never commit API keys.