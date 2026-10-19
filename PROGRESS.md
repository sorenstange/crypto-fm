# Progress

_Last updated: 2026-10-19, session 3. M3 criteria met by `cfm_bb_v1` (job 29758436) — diagnostics and controls done (no leak; gain is mostly volatility/tick size, direction ≈ short-term reversal) — **awaiting the user's decision on accepting M3**. Return heads overfit (best checkpoint at step 1000). No jobs running. Commits not pushed (no GitHub credentials on the cluster yet)._

## Current milestone
**M3 (backbone v1) — IN PROGRESS.** All code is written, committed, unit-tested and smoke-tested on an L40S. Full run `cfm_bb_v1` (job 29758436) **meets the M3 criteria** (below); reported to the user, not yet accepted.

Session 2 summary: M0 (data), M1 (cleaning + features) and M2 (baselines) were completed and accepted; M3 code was written.

M3 status:
- Written (commits a6c5f6f, caebe51, 17d682d): `configs/backbone.yaml`, `src/cfm/data/dataset.py`, `src/cfm/model/{blocks,backbone}.py`, `src/cfm/losses.py`, `src/cfm/eval/model_eval.py`, `src/cfm/train.py`, `jobs/train.sh`, `tests/test_model.py` (6 pass on the login node: v1 has 4.5–6M params, model causal, initial CE ≈ ln 66 and vol head = EWMA at init, losses match `eval.metrics`, every val point scored exactly once, √size symbol weights, sampler state round-trips).
- Smoke-test attempts (5 symbols BTC/ETH/SOL/DOGE/HYPER, 600 steps, run dir `data/runs/smoke_m3`, W&B group `smoke`):
  - 29755217 (gpua100): ImportError `seed_all` → moved it to `utils/checkpoint.py` (caebe51).
  - 29755221 (gpua100): got through the data build (**776,218 patches from 5 symbols in 6 s; 647,672 train window starts; 1,841 val windows**, A100 80GB), then `PermissionError: '/smoke_m3'` — `$RUNS_DIR` in an inline `bsub "..."` command string expanded to empty → pass an absolute path instead.
  - 29755223 (gpua100): killed while pending (~2,000 jobs queued on gpua100).
  - 29757779 (gpul40s, `-n 8`): killed while pending (no free L40S slot with 8 cores).
  - **29757820 (gpul40s, `-n 4`, 16 GB/core): PASSED** (167 s, L40S 46 GB, 3.1 GB host RAM; W&B `rudyhuy/crypto-fm/runs/afa7d897`; `data/runs/smoke_m3/` with COMPLETED, best/last checkpoints, eval/). Whole pipeline works: data build, compile, training, eval at 500/600, final paired eval with bootstrap, W&B.
- Smoke numbers (5 symbols, 600 steps, 117,833 val points — **not the M3 result**): initial CE 4.18–4.26 (ln 66 = 4.19), initial QLIKE 0.45 (= EWMA). Final: CE gain over marginal h1 0.189 [0.176, 0.201], h5 0.179, h15 0.173, h60 0.141 (all CIs > 0); rank IC pooled h1 0.097 / h5 0.055 / h15 0.046 / h60 0.037, cross-sectional 0.074 / 0.038 / 0.039 / 0.023; QLIKE h15 0.274 vs EWMA 0.672 vs HAR 0.280 (≈ HAR, CI spans 0), h60 0.277 vs HAR 0.267. Speed 1.52M patches/s = 46 steps/s at batch 256×128 (20k steps ≈ 8 min of training), GPU memory 2.9 GB for training.
- Session 3: `jobs/tests.sh` (job 29758411) **82 passed**; target alignment in `dataset.py` re-read (features up to the patch's last minute, targets from that minute on). The CE gain is plausibly mostly volatility information: z is scaled by EWMA σ (a poor vol forecast, QLIKE 0.67 vs HAR 0.28 on these symbols) and a Gaussian's CE excess from a wrong scale ≈ QLIKE/2 ≈ 0.2 nats. `jobs/train.sh` walltime 8 h → 2 h (38325ec); `data/runs/smoke_m3` deleted.
- **The CE gains / h1 IC look large for this SNR → run leakage/attribution diagnostics before trusting M3 numbers** (Next steps 1).

**M3 full run `cfm_bb_v1`** (job 29758436, L40S, 585 s total; W&B `rudyhuy/crypto-fm/runs/38dbd5b0`; `data/runs/cfm_bb_v1/eval/`):
- Data build: 31,916,062 patches × 15 × 17 from 370 symbols (16.3 GB fp16) in 193 s; 23.7M train window starts, 117,894 val windows. Peak host RAM 24.8 GB, GPU 19.6 GB, 1.53M patches/s.
- Initial CE 4.24/4.21/4.21/4.23 (ln 66 = 4.19); initial QLIKE 0.57/0.54 (= EWMA on the batch).
- **Early stop at step 6000; best val loss at step 1000 (still in LR warm-up).** Afterwards train CE keeps falling (h60 3.13 → 2.72 at step 6000) while val CE gets worse, most for long horizons (val CE gain h60 0.104 → −0.05, h15 0.136 → 0.04, h1 flat ≈ 0.20). Vol heads do not overfit (val QLIKE gain vs EWMA stays 0.24/0.22). Likely memorisation of market-wide moves: one time point's outcome is shared by ~300 correlated symbols and 4 overlapping h60 targets, and timestamps are identifiable from time encodings + context.
- **Final, best checkpoint on all 7,549,980 val points (95% day-bootstrap CI):** CE gain vs train marginal h1 0.219 [0.213, 0.225], h5 0.167 [0.162, 0.172], h15 0.138 [0.134, 0.143], h60 0.106 [0.101, 0.110]; QLIKE gain vs EWMA h15 0.246 [0.235, 0.256], h60 0.229 [0.219, 0.240]; **vs HAR h15 0.033 [0.029, 0.038], h60 0.017 [0.013, 0.020]** (already above the M5 bar); model QLIKE 0.247 / 0.235. Rank IC pooled / cross-sectional: h1 0.120 / 0.157, h5 0.066 / 0.083, h15 0.044 / 0.054, h60 0.030 / 0.028.
- Checkpoints: `best.pt` (step 1000) only (`last.pt` deleted after the diagnostics).

**M3 diagnostics** (`python -m cfm.eval.diagnostics`, job 29758508; `data/runs/cfm_bb_v1/eval/diagnostics{.json,_table.csv}`), best.pt on all 7.55M val points, CE in nats:
- Decomposition of the CE gain vs the marginal (h1 / h5 / h15 / h60): total 0.219 / 0.167 / 0.139 / 0.106; **HAR-rescaled marginal alone** (train ECDF of z/s, s = sqrt(HAR/EWMA variance)) 0.100 / 0.100 / 0.096 / 0.091 (CIs ±0.004); model over the HAR-rescaled marginal 0.119 / 0.067 / 0.043 / 0.015 (paired CI not computed). **Directional part** (CE of the mirror-symmetrised model minus model) 0.058 / 0.026 / 0.015 / 0.002, but on non-zero-return points only **0.002 / 0.001 / 0.001 / −0.003** — the rest is the exact-zero spike sitting in the bin above 0. ECDF marginal reproduces the M2 marginal exactly (sanity).
- **Tick size:** 10.3% of h1 val returns are exactly 0 (h60 1.9%); per-symbol zero share q10/50/90 = 1.6% / 6.0% / 23%, max CELO 47%, FLOW/GTC/LINA/FLM ~44%; those symbols get h1 CE gains ≈ 1 nat; per-symbol Spearman(zero share, gain) 0.59.
- **Rank IC vs simple reversal** (pooled / cross-sectional): model h1 0.120 / 0.157 vs −last 1-min return 0.078 / 0.139 (Spearman(model, rev1) 0.57); h15 model 0.044 / 0.054 vs −last patch return 0.036 / 0.066; h60 0.030 / 0.028 vs 0.025 / 0.043. Directional skill ≈ short-term reversal / bid-ask bounce; no evidence of more.

**M3 controls** (`jobs/controls.sh`, array 29758509, same config, stopped at 3,000 steps; W&B group `diagnostics`; checkpoints deleted, eval CSVs kept):
- `diag_shuffle` (train return targets permuted; W&B `890c2326`): val CE gain 0.0007 / 0.0013 / −0.0028 / 0.0001, IC ≈ 0 (≤ |0.03|) → **no leakage in the data/eval pipeline**. Its vol head is *better* than cfm_bb_v1's (QLIKE 0.240 / 0.228, vs HAR +0.041 / +0.024; best ckpt at step 2500) → the overfitting return heads hurt the shared backbone / force an early checkpoint.
- `diag_lag1` (inputs end one patch before the prediction point; W&B `2bb8dd51`): CE gain 0.177 / 0.143 / 0.125 / 0.103 (drop 0.042 / 0.024 / 0.014 / 0.003); IC cross-sectional 0.026 / 0.009 / 0.011 / 0.015 (from 0.157 / 0.083 / 0.054 / 0.028) → the directional signal lives in the last 15 minutes; vol head falls just below HAR (−0.006 / −0.004). Same overfitting pattern.

M2 evidence (`data/eval/baselines/summary.md`, job 29755134, 286 s; W&B `rudyhuy/crypto-fm/runs/2d668994`, group `baselines`):
- Prediction points = patch ends (:14/:29/:44/:59) after the 1440-min warm-up: train 24.0M, val 7.55M (val = 2025-01-04 … 2025-09-30 after the 3-day embargo). Test never touched.
- **Returns (CE in nats over 66 bins; ln 66 = 4.19):** the train marginal is the best baseline — h1 3.347, h5 3.393, h15 3.362, h60 3.348. Gaussian N(0,1) in z-space is 0.06–0.10 nats worse (tails), calibrated Gaussian 0.04–0.08 worse; CRPS nearly equal (marginal ≈ 0.61–0.64). Marginal PIT max decile deviation ≤ 0.023 (well calibrated). Rank IC n/a (constant forecasts).
- **Volatility (QLIKE, deseasonalised RV, all symbols):** EWMA h15 0.493 / h60 0.465; **HAR h15 0.280 / h60 0.252** (gain 0.21, CI 0.20–0.22); on the 273 GARCH-fitted symbols GARCH 0.335 / 0.335 sits between EWMA (0.419 / 0.393) and HAR (0.268 / 0.241). Train-calibrated EWMA is slightly worse than raw EWMA on val.
- Bars for later milestones: **M3** beat marginal CE (≈3.35–3.39) and EWMA QLIKE with CI excluding 0; **M5** beat HAR QLIKE (0.280 / 0.252).
- HAR's gain over EWMA is largest for low-liquidity symbols (0.33) and high-vol regimes (0.67).
- `jobs/tests.sh` (job 29755158): **76 passed**.

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

## Next steps (M3)
1. ~~Diagnostics~~ **done** (see above). Was: (the CE gains / h1 IC look large for this SNR): (a) a "HAR-scaled marginal" return baseline — marginal shape rescaled by sqrt(HAR/EWMA variance) — to separate the volatility-scale part of the CE gain; (b) the IC of a trivial 1-minute reversal signal (−last return) to see how much of the h1 IC is bid-ask bounce; (c) a shuffled-target control (permute targets across windows → gain must vanish); (d) a one-patch feature-lag control (input ends one patch before the prediction point → gain should shrink, not collapse); also check the share of the zero-return bin per symbol (tick-size discreteness on low-priced coins).
2. Full run done (job 29758436): all M3 criteria met. **Waiting for the user's verdict on accepting M3.** Proposed for later milestones: HAR-rescaled marginal as the return baseline and reversal signals as IC baselines (vs marginal is a weak bar).
3. Overfitting of the return heads (best at step 1000, during warm-up): options to discuss — eval every 100–250 steps, stronger regularisation (dropout, weight decay), lower LR, fewer/shorter-overlap targets, removing absolute-time cues, or a data-fraction check in the M4 ladder. Do not start M4 before this is decided.
4. Clean-up: `data/runs/smoke_m3` deleted (session 3); W&B smoke runs `afa7d897` (and the failed attempts) can be deleted if the user wants.
5. Push: needs GitHub credentials on the cluster (no gh, no SSH key, no credential helper) — user to choose a PAT push via `! git push https://sorenstange@github.com/...` or an SSH key for the cluster.
6. Later: consider dropping the `filled` feature (always 0).

## Decisions made
- Section 15 answers: recorded in plan.md §15.1 (universe top ~50–100 USDT perps from 2020; c = 0.05% + 0.02%; equal-weight portfolio; 15-min decisions).
- **Storage: 30 GB home quota only.** `DATA_ROOT=<project>/data` (gitignored), venv `<project>/.venv` (~9.9 GB). Delete irrelevant files continuously (caches, intermediates, superseded checkpoints). Check usage with `getquota_zhome.sh` (updates every ~4 h; `du -sh ~` is live).
- **Internet:** login, CPU (`hpc`) and GPU (`gpua100`) nodes all reach data.binance.vision, the S3 listing, fapi.binance.com, api.wandb.ai, PyPI directly (no proxy). → downloads run as batch jobs; W&B online.
- **W&B:** default entity of the logged-in account (`rudyhuy`, `WANDB_ENTITY` unset), project `crypto-fm`. Key in `~/.netrc`. Mode auto-detected in common.sh. Run ID stored in `<run_dir>/wandb_id.txt`; resumed jobs reuse it. All metrics logged against `trainer/step` (not W&B's internal step).
- **torch 2.14.1+cu130** (smaller than cu126 — chosen for space). No V100 support → don't use `gpuv100`; use `gpua100` / `gpul40s` / `gpuh100`. No CUDA module is loaded (wheel bundles the runtime).
- Queues: **GPU `gpul40s`** (user decision 2026-10-19: gpua100 had ~2,000 pending jobs; `gpuh100` is closed to this account — "User cannot use the queue"), 4 cores + 16 GB/core per GPU job (8-core requests did not fit on free L40S slots); CPU `hpc`; no LSF email.
- **Data/split decisions (session 2, details in plan.md §15.2–15.3):** maximise candles within the quota; universe floor $5M median daily quote volume; quote-volume columns dropped; splits train ≤ 2024-12-31 / val 2025-01…09 / test 2025-10…2026-09, 3-day embargo; no second data copy (segment index + in-RAM features); pooled seasonality; causal adaptive σ floor; scale σ_{t-1}·sqrt(season_t).
- **M3 design decisions (session 2):** features live on the GPU as fp16 (train + val ≈ 16 GB; test never loaded); the vol head predicts the log-ratio to the EWMA forecast (zero-init → starts at EWMA, scale-free); validation points are scored once each with ≥ 64 patches of context and compared to the baselines on identical points; best checkpoint by summed val loss (Σ CE + w_vol Σ QLIKE); heads live in `model/backbone.py` (no separate `heads.py`).
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
- None. (29758436 `cfm_bb_v1` finished DONE, COMPLETED at step 6000.)

## Code map (what exists)
- `env/setup_env.sh`, `env/modules.sh`, `env/requirements.txt` (+ `requirements.lock.txt`)
- `jobs/common.sh` (paths, venv, W&B env, `cfm_run`, `cfm_net_report`), `jobs/test_gpu.sh`, `jobs/check_net.sh`, `jobs/tests.sh`, `jobs/resume_test.sh`, `jobs/submit_chain.sh`
- `src/cfm/utils/checkpoint.py` (atomic save, RNG capture/restore, `seed_all`), `src/cfm/utils/run.py` (run info, `init_wandb`, `GracefulStop`) — reuse these in `cfm.train`
- `src/cfm/smoke/gpu_check.py`, `src/cfm/smoke/resume_test.py`; `tests/test_resume.py`
- `scripts/check_wandb_run.py`, `scripts/wandb_sync.sh` (offline fallback, currently unneeded)
- Eval (M2): `configs/data.yaml` `targets:` section; `src/cfm/data/splits.py` (`split_codes` with embargo, `prediction_points`), `targets.py` (z_h with scale σ_t·sqrt(Σ season), RV_h, 66 bins), `src/cfm/eval/metrics.py` (CE, binned CRPS, randomized PIT, rank IC, QLIKE, log-MAE, `day_bootstrap`), `baselines.py` (marginal, Gaussian, EWMA, log-HAR, GARCH(1,1) with stable h-step sums), `evaluate.py` (`python -m cfm.eval.evaluate baselines`); job `baselines.sh`; tests `test_targets_baselines.py`. Fitted params: `data/stats/baselines.json`.
- Backbone (M3, untrained): `configs/backbone.yaml` (model d256/L6/H4/SwiGLU 768/dropout 0.1; context 128 patches, warm-up 96; AdamW 1e-3 WSD 20k steps, batch 256, bf16, compile; eval every 500 on 2,048 val windows; early stop 10). `src/cfm/data/dataset.py` (`build_patch_data`: per-symbol features + targets cut into clock-aligned 15-candle patches, only data ≤ val_end (+1 day for targets) — test never loaded, asserted; `train_starts`, `start_weights` (√size), `WindowSampler` (numpy RNG, checkpointable), `eval_windows` (each val point once, ≥ 64 patches context)). `src/cfm/model/blocks.py` (RMSNorm, RoPE, QK-norm attention via SDPA, SwiGLU), `backbone.py` (patch embed → blocks → 4×66 return logits + 2 vol log-ratios to EWMA; zero-init vol head; no decay on norms/patch embed). `src/cfm/losses.py` (CE + z-loss, QLIKE as expm1(a)−a). `src/cfm/eval/model_eval.py` (`GPUData`, `predict_points`, `compare`: model vs marginal / EWMA / HAR on identical points, day bootstrap, rank IC). `src/cfm/train.py` (`python -m cfm.train --run-dir ...`; `--symbols`, `--total-steps`, `--max-minutes`, `--no-compile`); checkpoints `last.pt` + `best.pt` in `<run>/checkpoints/`, final eval in `<run>/eval/`. Job `jobs/train.sh` (gpul40s, run name = job name).
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
- BTC/ETH GARCH(1,1) on 1-min data is near-integrated (α+β ≈ 1); `garch_forecast` uses a stable sum and starts segments from the train sample variance. 23 symbols with train data got no GARCH (< 30 days in their last train year or invalid fit) → GARCH is compared on the 273-symbol subset.
- Home usage after M1: **quota 12.98 GB of 30 GB** (`getquota_zhome.sh`, 2026-10-19 ~14:00) while `du -sh ~` says 20 GB — the quota counts ~apparent size, not NFS block allocation. ~17 GB free. processed/ and stats/ are < 1 MB.
- Data facts from validation: no funding settlement exists at 2026-06-24 04:00 for 4h-interval symbols (missing in the bucket *and* REST → a real skipped settlement, 189 symbols); 4 single-symbol long funding gaps (≥20 days, halts). 20 candles in 19 symbols have taker_buy_base > volume (upstream rounding). LIT zero-volume share 16%.
- Raw store layout: `data/raw/parquet/<S>.parquet` (int-encoded; decode only via `cfm.data.load.read_klines`), `data/raw/funding/<S>.parquet`, `data/raw/manifest/<S>.csv` (per-symbol; needed for resume skip) + merged `manifest.csv`, `data/raw/universe/` (scan cache, 1d klines), `data/raw/validation/`.
- **Inline `bsub "..."` commands: `$RUNS_DIR` / `$DATA_ROOT` expand to empty** even after `source jobs/common.sh` in the same string → use absolute paths (expanded at submission) or a job script.
- GPU queue load 2026-10-19: gpua100 ~2,000 pending / 133 running; gpul40s ~150 pending / 80 running; **gpuh100: "User cannot use the queue"**. Free L40S slots often lack 8 cores → GPU jobs request `-n 4`.
- Smoke-test data build saw an **A100 80GB** node; gpua100 also has 40 GB nodes — on L40S (48 GB) the ~16 GB of fp16 features + model fit comfortably.
- GPU node RAM: gpua100 hosts 755 GB–1 TB (`lshosts`); L40S node RAM not yet checked (job requests 64 GB).
- **`git push` fails on the cluster:** no GitHub credentials (HTTPS prompts for a username and cannot; no `gh`, no `~/.ssh` key). origin = https://github.com/sorenstange/crypto-fm.git.
- **One Claude session at a time:** in session 3 a leftover background session (from session 2, under the `claude daemon`) committed to PROGRESS.md concurrently. Check `ps -u $USER -f | grep claude` / ListAgents at session start; close stray ones.
- `/dtu/blackhole` and `/work3` need HPC-support-created dirs; the user chose not to request them.

## How to resume work
```bash
cd ~/crypto-fm
bjobs -a                       # anything still running?
bsub < jobs/tests.sh           # sanity: unit tests
source jobs/common.sh          # login-node env for light work (wandb API, etc.)
```
