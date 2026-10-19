# Progress

_Last updated: 2026-10-19, session 2 (M1 done). No jobs running._

## Current milestone
**M1 (cleaning + features) — DONE 2026-10-19.** Next: M2 (baselines, plan.md §11) or Phase 3 dataset/targets — wait for the user's go-ahead.

M1 evidence:
- `jobs/clean.sh` (job 29754839, 89 s): 641.3M raw rows → **638.5M kept (99.58%) in 3,688 segments**; dropped: listing warm-up 1.61M, identical runs 0.73M, short segments 0.38M (zero-volume runs are already covered by identical runs); breaks: 5,541 jumps > 10%, 105 long gaps; 0 fills needed. Checks all 0 (NaN, non-1-min steps, OHLC, taker > volume). Report: `data/processed/clean_report.csv`, `clean_summary.txt` (incl. survivorship: 81 included symbols no longer trade; 80 delisted/settling ones excluded only by history/liquidity rules).
- `jobs/features.sh` (job 29754863): pooled seasonality from 296 symbols (factor 0.48…3.81), scalers lnsig (−6.585, 0.772), dvol (9.671, 1.723) → `data/stats/`. Features for all 371 symbols in 119 s wall (max 4 s/symbol), **0 non-finite**, 43.4 GB at float32 in RAM. Train moments: body std 0.98, ret std 1.02 (normalisation works); val/test drift in dvol (−0.4 / −1.1) and funding (lower).
- `jobs/tests.sh` (job 29754864): **66 passed** — causality tests (future perturbation incl. funding, at segment boundaries / fills; synthetic and real BTC/LUNA with real stats), candle validity (synthetic + 4 real symbols), cleaning rules, encoding, bucket client.
- Funding backfill from REST: ICPUSDT 1,433 rows (2021-05…2022-08), BNXUSDT 1,004 (2022-04…2023-01; incl. real 2h funding in Dec 2022), QTUMUSDT 28.

M0 evidence (`data/raw/validation/summary.txt`, job 29754547):
- Universe: **371 symbols** (`configs/universe.yaml`; COIN-underlying USDT perps, ≥365 active days, median daily quote volume ≥ $5M, delisted/settling included).
- Download array 29754541 (371 elements, ~30 min): **31,063 files, all `ok`, every one SHA-256-verified** (15,860 kline months + 15,203 funding months, up to 2026-09). Merged manifest `data/raw/manifest.csv`.
- **641,266,398 candles**, kline parquet 6.6 GB apparent (10.3 B/candle), funding 16 MB.
- Coverage median 1.0000 (min 0.975 BNX); 0 duplicates, 0 non-monotonic, 0 OHLC-invalid, 0 NaN; 105 long gaps (all ≥ 3 days: halts, delist/relist), none ≤ 15 min.
- REST spot-check: 60 symbol-days (BTC, ETH, ENS, CHR, FIDA, RVN) — **0 mismatches**, identical row sets.
- Funding present for every symbol; intervals 1/2/4/8 h.

M−1 evidence:
- `jobs/test_gpu.sh` (job 29753877, A100-PCIE-40GB): bf16 fwd/bwd, initial CE ≈ ln(66), flash SDPA, torch.compile all PASS.
- `jobs/tests.sh` (job 29754141): 6/6 pytest pass (CPU resume is bitwise identical).
- Chained resume test `cfm_resume_10191056` (jobs 29754139 → 29754140): one continuous W&B run `rudyhuy/crypto-fm/runs/334b5f4a`.

## Next steps (after user go-ahead)
1. M2 baselines (plan.md §11): marginal over bins, EWMA-Gaussian, HAR-RV, GARCH(1,1) on validation; needs Phase 4 targets (binned returns at h ∈ {1,5,15,60} scaled by `scale` = σ_{t−1}·sqrt(season_t), realised variance) → write `targets.py` first, then `baselines.py`, `metrics.py` with day-block bootstrap.
2. Phase 3 dataset (`splits.py`, `dataset.py`): in-RAM features per symbol (43 GB fp32 / 22 GB fp16; gpua100 nodes have 755 GB–1 TB RAM), √size symbol weights, patches P=15 clock-aligned, 128-patch context + 1440 warm-up, embargo 3 days, optional patch-anchored normalisation.
3. Consider dropping the `filled` feature (always 0: there are no gaps ≤ 15 min in the data).

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

- Universe is ranked on the **active** history: 52+ settled/delisted contracts keep producing flat zero-volume candles after trading ends (DAR: 927k after 2024-12-26); stats exclude that tail and the converter trims trailing zero-volume runs ≥ 1 day. Applying this moved the $5M-floor universe from 347 to 371 symbols (~642M est. candles).
- Pilot bytes/candle (int store, full history): BTC 14.4, ETH 13.2, LUNA 11.8, HYPER 9.3; full universe 10.3.

## Running / pending jobs
- None (`bjobs` empty at session end, 2026-10-19).

## Code map (what exists)
- `env/setup_env.sh`, `env/modules.sh`, `env/requirements.txt` (+ `requirements.lock.txt`)
- `jobs/common.sh` (paths, venv, W&B env, `cfm_run`, `cfm_net_report`), `jobs/test_gpu.sh`, `jobs/check_net.sh`, `jobs/tests.sh`, `jobs/resume_test.sh`, `jobs/submit_chain.sh`
- `src/cfm/utils/checkpoint.py` (atomic save, RNG capture/restore), `src/cfm/utils/run.py` (run info, `init_wandb`, `GracefulStop`) — reuse these in `cfm.train`
- `src/cfm/smoke/gpu_check.py`, `src/cfm/smoke/resume_test.py`; `tests/test_resume.py`
- `scripts/check_wandb_run.py`, `scripts/wandb_sync.sh` (offline fallback, currently unneeded)
- Data (M1): `configs/data.yaml` (splits, clean/vol/feature params), `src/cfm/data/config.py`, `clean.py` (segment index + `load_clean`), `vol.py` (variance estimators, pooled seasonality, per-segment EWMA, adaptive floor), `features.py` (`build_features`, `fit-stats` / `check` CLI, `load_stats`); jobs `clean.sh`, `features.sh`; tests `test_clean.py`, `test_features.py`, `test_real_data.py`. Funding backfill: `python -m cfm.data.download --backfill-funding --symbols ...`.
- Data (M0): `src/cfm/data/binance.py` (bucket listing/XML, HTTP retries, checksum, CSV parsing with header + ts-unit detection), `universe.py` (`scan` / `select`), `download.py` (per-symbol in-memory verified download → convert), `convert.py` (lossless int parquet, dedup, stale-tail trim, funding minute rounding), `load.py` (`read_klines`, `read_funding`), `validate.py`; jobs `universe.sh`, `download.sh` (array), `validate.sh`; tests `test_binance.py`, `test_convert.py` (30 pass)

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
- Home usage after M1: **quota 12.98 GB of 30 GB** (`getquota_zhome.sh`, 2026-10-19 ~14:00) while `du -sh ~` says 20 GB — the quota counts ~apparent size, not NFS block allocation. ~17 GB free. processed/ and stats/ are < 1 MB.
- Data facts from validation: no funding settlement exists at 2026-06-24 04:00 for 4h-interval symbols (missing in the bucket *and* REST → a real skipped settlement, 189 symbols); 4 single-symbol long funding gaps (≥20 days, halts). 20 candles in 19 symbols have taker_buy_base > volume (upstream rounding). LIT zero-volume share 16%.
- Raw store layout: `data/raw/parquet/<S>.parquet` (int-encoded; decode only via `cfm.data.load.read_klines`), `data/raw/funding/<S>.parquet`, `data/raw/manifest/<S>.csv` (per-symbol; needed for resume skip) + merged `manifest.csv`, `data/raw/universe/` (scan cache, 1d klines), `data/raw/validation/`.
- `/dtu/blackhole` and `/work3` need HPC-support-created dirs; the user chose not to request them.

## How to resume work
```bash
cd ~/crypto-fm
bjobs -a                       # anything still running?
bsub < jobs/tests.sh           # sanity: unit tests
source jobs/common.sh          # login-node env for light work (wandb API, etc.)
```
