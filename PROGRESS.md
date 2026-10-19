# Progress

_Last updated: 2026-10-19, session 2 (M0 step 1: bucket explored). No jobs running._

## Current milestone
**M−1 (HPC environment) — DONE and verified 2026-10-19.** M0 (data acquisition) is next; nothing for M0 written yet.

M−1 evidence:
- `jobs/test_gpu.sh` (job 29753877, A100-PCIE-40GB): bf16 fwd/bwd, initial CE ≈ ln(66), flash SDPA, torch.compile all PASS.
- `jobs/tests.sh` (job 29754141): 6/6 pytest pass (CPU resume is bitwise identical).
- Chained resume test `cfm_resume_10191056` (jobs 29754139 → 29754140): job 1 stopped on the LSF SIGUSR2 walltime warning at step 577, job 2 resumed to 600; uninterrupted rerun identical (loss/param diff 0.0); one continuous W&B run `rudyhuy/crypto-fm/runs/334b5f4a` (`scripts/check_wandb_run.py` PASSED).

## Next steps (M0, plan.md §2)
1. ~~Explore the bucket~~ — DONE 2026-10-19 (session 2), findings below under "Binance bucket facts".
2. `src/cfm/data/universe.py` — scan DONE (job 29754392, 11 min; `select` re-run on login node after a YAML numpy-float fix). 567 eligible symbols (145 not trading now), ~874M est. candles, 27.2 GB of 1m zips; ranked list in `configs/universe.yaml`, raw scan in `data/raw/universe/`. **Waiting for the user: liquidity floor / candle budget and whether to drop the quote-volume columns** (see Storage budget).
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

## Binance bucket facts (explored 2026-10-19, session 2)
- **Listing:** `https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?prefix=...&delimiter=/&marker=...`, 1000 entries/page, continue with `NextMarker` (or last `<Key>` when no delimiter). **Parse the XML properly** — newer `<Contents>` entries carry extra `<ChecksumAlgorithm>/<ChecksumType>` elements; a naive Key…ETag…Size regex silently dropped everything after 2025-01.
- futures/um monthly types: aggTrades, bookTicker, fundingRate, indexPriceKlines, klines, markPriceKlines, premiumIndexKlines, trades. Daily additionally: bookDepth, metrics.
- **Symbols (monthly/klines): 1056** = 900 `*USDT`, 52 dated (`_YYMMDD`), 40 `*USDC`, 41 `*BUSD`, plus odd ones: `ETHBTC`, `BTCU`/`ETHU`, `*USD1`, and 17 `*USDTSETTLED` / `*USDT_SETTLED` (even `AERGOUSDTSETTLEDSETTLED`). All 900 USDT symbols have a fundingRate dir.
  - `*SETTLED` = copies/fragments of a contract around a settlement/delist-relist (ICPUSDT_SETTLED duplicates ICPUSDT 2022; BNXUSDTSETTLED = old pre-2023-02-22 BNX contract at ~100 USDT, while BNXUSDT relisted redenominated at ~1.4 and ends the old stretch with flat zero-volume candles). → **exclude `*SETTLED`**; the main symbol's long gaps / price jumps are handled by Phase-1 segment splitting.
  - Index/basket: `BTCDOMUSDT`, `DEFIUSDT`, `BLUEBIRDUSDT`, `FOOTBALLUSDT`. Stable/commodity: `USDCUSDT`, `PAXGUSDT`, `XAUTUSDT` (exclude). `BUSDT`/`TUSDT` are regular tokens (B, T), not stablecoins.
- **Coverage:** BTCUSDT 1m monthly 2020-01 … 2026-09 (81 files, 152 MB zipped, ~1.9 MB/month); daily files cover the current month up to yesterday (2026-10-18). premiumIndex/markPrice klines from 2020-01; daily `metrics` from 2020-09.
- **Kline CSV:** 12 columns (open_time, open, high, low, close, volume, close_time, quote_volume, count, taker_buy_volume, taker_buy_quote_volume, ignore). **Header row varies:** absent in older files (BTCUSDT 2020-01, ICPUSDT 2022-03), present in others (BTC 2022-06+, ICPUSDT_SETTLED 2022-03) → detect per file. Timestamps are **13-digit ms in every futures file checked (incl. 2026 daily)**; keep the magnitude check anyway.
- **Checksum:** `<sha256>  <filename>` in `<url>.CHECKSUM`; verified OK on all samples.
- **fundingRate CSV:** header `calc_time,funding_interval_hours,last_funding_rate`; `calc_time` in ms with **0–20 ms jitter** (e.g. …800005) → round to the nearest minute. Interval stored per row: 8h (BTC, DOGE, PEPE), 4h (WIF, TRUMP), 1h (ALPACA 2025-06). No mark price column.
- 1d klines exist per symbol/month (`klines/{S}/1d/`, ~4 KB each) for the volume ranking; a listing request takes ~1.5 s.
- Rough size: 100M candles ≈ 2,300 symbol-months ≈ ≤ 4.4 GB of zips at BTC's per-month size (alts likely smaller).

## Storage budget (M0 step 2, 2026-10-19)
- User wants **as many candles as fit**, leaving headroom for models and other tasks (not just ≥100M).
- Parquet bytes/candle measured on 1-month samples: float64 + zstd ≈ 43–50 B; **lossless int encoding** (prices as int ticks, volumes as int lots, DELTA_BINARY_PACKED + zstd) ≈ 26 B; same without `quote_volume`/`taker_buy_quote` ≈ 15 B (those two are ~half the bytes; ≈ volume×price).
- Free ≈ 19 GB. Proposed reserve ≈ 8.5 GB (checkpoints 3, W&B/logs/eval 1, stats/indices 1, in-flight zips 1.5, safety 2) → **~10.5 GB for raw klines**. Assumes Phase 1/2 don't materialise a second full copy (cleaning → segment index; features computed in RAM at load time or stored compactly).
- Floors (median daily quote volume): ≥$5M → 348 symbols, ~648M candles, ~9.7 GB @15 B; ≥$10M → 218 / 476M; ≥$2M → 482 / 781M / 11.7 GB.
- venv is 9.9 GB (nvidia 4.2, torch 2.1, triton 1.2); not touched.

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
