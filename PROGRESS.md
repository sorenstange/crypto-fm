# Progress

_Last updated: 2026-10-19, end of session 1 (M−1). All work committed; no jobs running._

## Current milestone
**M−1 (HPC environment) — DONE and verified 2026-10-19.** M0 (data acquisition) is next; nothing for M0 written yet.

M−1 evidence:
- `jobs/test_gpu.sh` (job 29753877, A100-PCIE-40GB): bf16 fwd/bwd, initial CE ≈ ln(66), flash SDPA, torch.compile all PASS.
- `jobs/tests.sh` (job 29754141): 6/6 pytest pass (CPU resume is bitwise identical).
- Chained resume test `cfm_resume_10191056` (jobs 29754139 → 29754140): job 1 stopped on the LSF SIGUSR2 walltime warning at step 577, job 2 resumed to 600; uninterrupted rerun identical (loss/param diff 0.0); one continuous W&B run `rudyhuy/crypto-fm/runs/334b5f4a` (`scripts/check_wandb_run.py` PASSED).

## Next steps (M0, plan.md §2)
1. Explore the bucket before writing parsers: futures/um listing (paginated), `fundingRate` path + columns, header row present or not, timestamp unit (ms vs µs) per file.
2. `src/cfm/data/universe.py`: list all `*USDT` perps incl. delisted, apply §2.2 exclusions, rank by median daily quote volume (1d klines), ≥12 months history. **Show the symbol list to the user for review** before the full download.
3. Pilot: `download.py` + `jobs/download.sh` (CPU queue `hpc`; compute nodes have internet) for BTCUSDT/ETHUSDT, 2 months; checksum-verified, manifest written.
4. **Storage budget** from the pilot sizes: extrapolate zips + parquet for the full universe against the free quota (~19 GB) before the full download. Delete zips once converted + verified (keep manifest with sha256).
5. Full download (job array per symbol), `convert.py`, `validate.py`, REST spot-checks → M0 acceptance. Then stop and report.

## Decisions made
- Section 15 answers: recorded in plan.md §15.1 (universe top ~50–100 USDT perps from 2020; c = 0.05% + 0.02%; equal-weight portfolio; 15-min decisions).
- **Storage: 30 GB home quota only.** `DATA_ROOT=<project>/data` (gitignored), venv `<project>/.venv` (~9.9 GB). Delete irrelevant files continuously (caches, intermediates, superseded checkpoints). Check usage with `getquota_zhome.sh` (updates every ~4 h; `du -sh ~` is live).
- **Internet:** login, CPU (`hpc`) and GPU (`gpua100`) nodes all reach data.binance.vision, the S3 listing, fapi.binance.com, api.wandb.ai, PyPI directly (no proxy). → downloads run as batch jobs; W&B online.
- **W&B:** default entity of the logged-in account (`rudyhuy`, `WANDB_ENTITY` unset), project `crypto-fm`. Key in `~/.netrc`. Mode auto-detected in common.sh. Run ID stored in `<run_dir>/wandb_id.txt`; resumed jobs reuse it. All metrics logged against `trainer/step` (not W&B's internal step).
- **torch 2.14.1+cu130** (smaller than cu126 — chosen for space). No V100 support → don't use `gpuv100`; use `gpua100` / `gpul40s` / `gpuh100`. No CUDA module is loaded (wheel bundles the runtime).
- Queues: GPU `gpua100`, CPU `hpc`, no LSF email.
- Session workflow (CLAUDE.md): read PROGRESS.md + plan.md at session start; update PROGRESS.md and commit before ending.
- Commits go directly on `main`. Remote `origin` = github.com/sorenstange/crypto-fm; session-1 commits are **not pushed** (push only when the user asks).
- Chaining: `jobs/submit_chain.sh N script name` chains by **job ID** (`-w done(id)`); all jobs share the job name → same run dir `$RUNS_DIR/<name>`. `DEP=ended` continues after crashes.
- Walltime: `#BSUB -wa USR2 -wt <min>` + `cfm_run` (common.sh) forwards SIGUSR2/SIGTERM to Python; `cfm.utils.run.GracefulStop` → checkpoint and exit 0. Use `-wt 15` for real training.

## Running / pending jobs
- None (`bjobs` empty at session end, 2026-10-19).

## Code map (what exists)
- `env/setup_env.sh`, `env/modules.sh`, `env/requirements.txt` (+ `requirements.lock.txt`)
- `jobs/common.sh` (paths, venv, W&B env, `cfm_run`, `cfm_net_report`), `jobs/test_gpu.sh`, `jobs/check_net.sh`, `jobs/tests.sh`, `jobs/resume_test.sh`, `jobs/submit_chain.sh`
- `src/cfm/utils/checkpoint.py` (atomic save, RNG capture/restore), `src/cfm/utils/run.py` (run info, `init_wandb`, `GracefulStop`) — reuse these in `cfm.train`
- `src/cfm/smoke/gpu_check.py`, `src/cfm/smoke/resume_test.py`; `tests/test_resume.py`
- `scripts/check_wandb_run.py`, `scripts/wandb_sync.sh` (offline fallback, currently unneeded)

## Gotchas / open issues
- The python3 module sets `PYTHONPATH`/`PYTHONHOME` to its site-packages, which shadows the venv; `cfm_load_modules` unsets both. Don't re-add them.
- Env vars set at `bsub` time (e.g. `DATA_ROOT=... bsub`) did **not** reach the job; put settings in job scripts.
- Always `torch.load` checkpoints to CPU (`load_checkpoint` default); RNG/generator states must be CPU ByteTensors.
- wandb 0.30 has no `wandb.util.generate_id` (we use `secrets.token_hex`).
- `gpua100` max runlimit is actually 72 h (plan assumes 24 h; design kept at 24 h). Default runlimit is 15 min if `-W` is omitted.
- GPU nodes report driver 615 / CUDA 13.4 (bhosts showed 580 for V100 nodes).
- resume_test margin was thin (SIGUSR2 at step 577/600); raise `--total-steps` if it is rerun.
- `.venv` also contains the user's own unrelated packages (cwevals, opentelemetry, …); don't recreate the venv without asking.
- A crashed W&B run `f193c8f3` from the first resume attempt is still in the project (user may delete it).
- Home usage at session end: 11 GB of 30 GB (`du -sh ~`), of which the venv is 9.9 GB and `data/` 3.5 MB.
- `/dtu/blackhole` and `/work3` need HPC-support-created dirs; the user chose not to request them.

## How to resume work
```bash
cd ~/crypto-fm
bjobs -a                       # anything still running?
bsub < jobs/tests.sh           # sanity: unit tests
source jobs/common.sh          # login-node env for light work (wandb API, etc.)
```
