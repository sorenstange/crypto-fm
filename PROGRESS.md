# Progress

_Last updated: 2026-10-19 ~20:40, end of session 3. **M3 accepted.** Pre-M4 regularisation fix done (plan.md §16): backbone **v2** config = v1 + lr 3e-4, return weights 1/1/0.5/0.25, sign-flip augmentation, 3,000-step schedule (val loss −0.108 vs v1). **Two jobs pending at session end: 29759200 `cfm_bb_v2` (full v2 run) and 29759201 (diagnostics on it, chained).** Commits not pushed (no GitHub credentials on the cluster)._

## Current milestone
**M3 (backbone v1) — DONE, accepted by the user 2026-10-19 (session 3)**, with the caveats below (gain mostly volatility + tick size; direction ≈ short-term reversal).
**Pre-M4 work (agreed order):** (1) regularisation fix — **done**, v2 run pending; (2) stronger yardsticks (HAR-rescaled marginal as the return baseline, reversal signals as IC baselines) — not started; then **M4 scaling ladder** (implementation decisions in plan.md §10.1).

### Session 3 summary
- M3 smoke test passed, full run `cfm_bb_v1` met all M3 criteria; diagnostics + shuffle/lag controls: **no leakage**, CE gain mostly volatility scale and tick size, directional part ≈ short-term reversal → **M3 accepted**.
- Overfitting of the return heads (best step 1,000) analysed in three sweeps: standard regularisers don't help (data-limited: onset ≈ 1 pass over the train patches); **sign-flip augmentation doubles the onset and gives −0.108 val loss**; 3,000-step schedule; seed noise ≈ 0.01. → v2 defaults in `configs/backbone.yaml`, v1 frozen in `configs/backbone_v1.yaml`.
- New infrastructure: `cfm.eval.diagnostics`, control flags in `cfm.train`, `--set` overrides, `cfm.sweep`, `cfm.eval.compare_runs` (paired CIs between runs), `cfm.data.augment`, HL-Gauss/ret-weight/feature-drop options.
- plan.md: §16 (regularisation), §10.1 (ladder implementation decisions), §17 (candidate data experiments after M4: lower universe floor, FX, stocks, macro calendar).
- A stray background Claude session from session 2 committed concurrently (868e7c3); the user closed it.

### M3 history (session 2–3)
- Code (a6c5f6f, caebe51, 17d682d): dataset, backbone, losses, model eval, train loop, `jobs/train.sh`, `tests/test_model.py`.
- Smoke attempts: 29755217 (ImportError `seed_all`), 29755221 (`$RUNS_DIR` empty in inline bsub), 29755223 / 29757779 (killed while pending); **29757820 passed** (5 symbols, 600 steps, 167 s; 1.52M patches/s ≈ 46 steps/s; W&B `afa7d897`). Unit tests: 82 passed (29758411).

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

**Pre-M4 regularisation (plan.md §16; configs/sweeps/reg1–3.yaml on `configs/backbone_v1.yaml`; all paired vs reg1/ref = v1 on all 7.55M val points, val loss = Σ4 CE + Σ2 QLIKE, positive = better):**
- Infrastructure (fd9e221, e10a0dc): `--set`, `loss.ret_weights`, `loss.hl_gauss_sigma` (soft targets centred on the hard bin's centre — exact-zero returns sit on the edge at 0), `model.drop_features`, `train.flip_prob` (`cfm/data/augment.py`, verified against features of a mirrored price path), `cfm.sweep`, `eval.save_points` + `cfm.eval.compare_runs`. CPU smoke `jobs/sweep_smoke.sh` (29758780) passed; full tests 92 passed (29758786; +1 flip test since, passes).
- **Round 1** (array 29758785; one change each): lr3e4 +0.014 [0.004, 0.022], retw (1/1/0.5/0.25) +0.008, nocal 0.000, drop03 −0.012, short −0.015, wd03 −0.025, drop02 −0.025, hlgauss −0.146 (blurs tick-size spikes, −0.086 at h1). **Best step 750–1250 in every run** → onset ≈ 1.4 passes over the 24.2M train patches: data-limited, not a regulariser problem. `ref` reproduced cfm_bb_v1 exactly (deterministic).
- **Round 2** (array 29759029; on lr3e4 + retw): **flip_ep2 +0.108 [0.100, 0.115]** (flip p 0.5, 3,000 steps, decay last 50%, best = 3000), **flip +0.105** (20k schedule, best 3250, early stop 6250), ep1 +0.062 (1,500 steps), base +0.041, ep1b +0.039. flip_ep2: CE h1/h5/h15/h60 +0.048/+0.023/+0.013/+0.008, QLIKE +0.0087/+0.0065 → vs HAR 0.042/0.023 (the level the shuffle control predicted); cross-sectional IC h1 0.147 (v1 0.157), h60 0.030. `data/runs/reg2/compare_vs_reg1.csv`.
- **Round 3** (array 29759168; vs flip_ep2): t4k −0.004, seed 1 −0.009, t5k −0.014 (both best at 2750) → **3,000 steps kept**; **seed-to-seed noise ≈ 0.01**, larger than the paired CIs. `data/runs/reg3/compare.csv`.
- **v2 = reg2/flip_ep2 exactly** (7ed5291; checked programmatically): `configs/backbone.yaml` lr 3e-4, ret_weights 1/1/0.5/0.25, flip_prob 0.5, total 3,000, warm-up 300, decay_frac 0.5, eval every 250, early stop 12, save_points true. Expected v2 numbers (= flip_ep2): val loss 13.1945; CE gain vs marginal 0.2669 / 0.1900 / 0.1518 / 0.1143; QLIKE vs HAR +0.0421 / +0.0230; IC cross 0.147 / 0.074 / 0.051 / 0.030.
- Per-point files kept: `data/runs/reg1/ref`, `data/runs/reg2/flip_ep2` (~190 MB each); all sweep checkpoints deleted.

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

## Next steps
1. **Check the v2 jobs.** `bjobs -a 29759200 29759201`; `grep -E "^\[(train|eval|final|diag)\]" logs/29759200.out logs/29759201.out`. Expect v2 to reproduce reg2/flip_ep2 (numbers above); confirm with a CPU job `python -m cfm.eval.compare_runs $PWD/data/runs/reg2/flip_ep2 $PWD/data/runs/cfm_bb_v2 --ref flip_ep2 --out $PWD/data/runs/cfm_bb_v2/compare_vs_flip_ep2.csv` (gain ≈ 0). If the jobs never ran or failed: `J=$(bsub -J cfm_bb_v2 < jobs/train.sh | grep -o "[0-9]\+" | head -1); bsub -w "done($J)" -J cfm_diag_cfm_bb_v2 < jobs/diagnostics.sh` (delete a partial `data/runs/cfm_bb_v2` first unless resuming).
2. **Read the v2 diagnostics** (`data/runs/cfm_bb_v2/eval/diagnostics.json`) vs v1's (`data/runs/cfm_bb_v1/eval/diagnostics.json`): HAR-scaled-marginal share of the CE gain, directional part on non-zero points, zero-return share, IC vs reversal. Report v2 to the user as the new reference for M4.
3. **Stronger yardsticks (pre-M4 step 2, agreed with the user):** add the HAR-rescaled marginal (`cfm.eval.diagnostics.ecdf_bin_probs`, ECDF of z/s fitted on train) as a paired return baseline in `cfm.eval.model_eval.compare` / the final table, and reversal signals (−last 1-min return, −last patch return) as IC baselines in the summaries. Re-evaluate v1 and v2 with it. Then stop and report.
4. **M4 scaling ladder** per plan.md §10 + §10.1: day-subset masks for 25/50/100% data in `train_starts` (one data build), schedule ∝ data fraction (~2 passes with flips), LR check at XS and XL, CE/QLIKE reported separately, ≥ 2 seeds where differences < ~0.01; a `configs/sweeps/ladder.yaml` + `jobs/sweep.sh`. Plan it with the user before running.
5. After M4: plan.md §17 data experiments, gated on the 50% → 100% data curve.
6. Housekeeping: delete the per-point files in `data/runs/reg1/ref` and `reg2/flip_ep2` once v2 is confirmed (v2 has its own); W&B smoke/diagnostic runs (`afa7d897`, `f193c8f3`, `diagnostics` group, `sweep_smoke` offline runs were never synced) can be deleted if the user wants.
7. Push: needs GitHub credentials on the cluster (no gh, no SSH key) — user to choose a PAT push (`! git push https://sorenstange@github.com/sorenstange/crypto-fm.git main`) or an SSH key.
8. Later: drop the `filled` feature (always 0); optional symbol-subsampling ladder variant; macro-calendar feature for M5 (§17.4).

## Decisions made
- Section 15 answers: recorded in plan.md §15.1 (universe top ~50–100 USDT perps from 2020; c = 0.05% + 0.02%; equal-weight portfolio; 15-min decisions).
- **Storage: 30 GB home quota only.** `DATA_ROOT=<project>/data` (gitignored), venv `<project>/.venv` (~9.9 GB). Delete irrelevant files continuously (caches, intermediates, superseded checkpoints). Check usage with `getquota_zhome.sh` (updates every ~4 h; `du -sh ~` is live).
- **Internet:** login, CPU (`hpc`) and GPU (`gpua100`) nodes all reach data.binance.vision, the S3 listing, fapi.binance.com, api.wandb.ai, PyPI directly (no proxy). → downloads run as batch jobs; W&B online.
- **W&B:** default entity of the logged-in account (`rudyhuy`, `WANDB_ENTITY` unset), project `crypto-fm`. Key in `~/.netrc`. Mode auto-detected in common.sh. Run ID stored in `<run_dir>/wandb_id.txt`; resumed jobs reuse it. All metrics logged against `trainer/step` (not W&B's internal step).
- **torch 2.14.1+cu130** (smaller than cu126 — chosen for space). No V100 support → don't use `gpuv100`; use `gpua100` / `gpul40s` / `gpuh100`. No CUDA module is loaded (wheel bundles the runtime).
- Queues: **GPU `gpul40s`** (user decision 2026-10-19: gpua100 had ~2,000 pending jobs; `gpuh100` is closed to this account — "User cannot use the queue"), 4 cores + 16 GB/core per GPU job (8-core requests did not fit on free L40S slots); CPU `hpc`; no LSF email.
- **Data/split decisions (session 2, details in plan.md §15.2–15.3):** maximise candles within the quota; universe floor $5M median daily quote volume; quote-volume columns dropped; splits train ≤ 2024-12-31 / val 2025-01…09 / test 2025-10…2026-09, 3-day embargo; no second data copy (segment index + in-RAM features); pooled seasonality; causal adaptive σ floor; scale σ_{t-1}·sqrt(season_t).
- **M3 design decisions (session 2):** features live on the GPU as fp16 (train + val ≈ 16 GB; test never loaded); the vol head predicts the log-ratio to the EWMA forecast (zero-init → starts at EWMA, scale-free); validation points are scored once each with ≥ 64 patches of context and compared to the baselines on identical points; best checkpoint by summed val loss (Σ CE + w_vol Σ QLIKE); heads live in `model/backbone.py` (no separate `heads.py`).
- **M3 accepted (user, session 3)** on the M3 criteria plus diagnostics/controls; caveat recorded: CE vs the train marginal is a weak bar (vol scale alone gives 0.09–0.10 nats).
- **Pre-M4 order (user):** regularisation fix first, then stronger yardsticks (HAR-rescaled marginal, reversal IC baselines), then M4.
- **Regularisation (user):** round-2 choice (a) one-epoch schedule + (b) sign-flip augmentation; market-relative return targets deferred (would make M7 market-neutral); separate vol checkpoint kept in mind for M5. **v2 = reg2/flip_ep2** becomes the default config; v1 frozen as `configs/backbone_v1.yaml` (base of reg1–3).
- **Sign-flip semantics:** mirror the price path (negate body/ret/taker_imb/funding/range_pos, swap upper/lower, z → −z), keep volume/vol/calendar features (dvol kept by design).
- **HL-Gauss soft targets** are centred on the hard bin's centre (not on z) — kept as an option, lost clearly in round 1.
- **Ladder implementation (user, plan.md §10.1):** data fractions by time (random days), schedule ∝ data, LR check at the extremes, returns/vol reported separately.
- **Cross-market data (user, plan.md §17):** not now; lower universe floor → FX → stocks, gated on the M4 data curve; macro event calendar as a separate vol-feature experiment.
- Diagnostics / controls / sweeps run as parallel GPU jobs (user declined merging them into one job to save queue slots).
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
- **29759200** `cfm_bb_v2` (gpul40s, 2 h walltime) — PENDING at session end → `data/runs/cfm_bb_v2/` (checkpoints kept, eval + val_points.npz), W&B group `backbone`.
- **29759201** `cfm_diag_cfm_bb_v2` (gpul40s, 1 h) — waits for `done(29759200)` → `data/runs/cfm_bb_v2/eval/diagnostics{.json,_table.csv}`. If 29759200 fails, this one never starts (`bkill 29759201` and resubmit both, see Next steps 1).

## Code map (what exists)
- `env/setup_env.sh`, `env/modules.sh`, `env/requirements.txt` (+ `requirements.lock.txt`)
- `jobs/common.sh` (paths, venv, W&B env, `cfm_run`, `cfm_net_report`), `jobs/test_gpu.sh`, `jobs/check_net.sh`, `jobs/tests.sh`, `jobs/resume_test.sh`, `jobs/submit_chain.sh`
- `src/cfm/utils/checkpoint.py` (atomic save, RNG capture/restore, `seed_all`), `src/cfm/utils/run.py` (run info, `init_wandb`, `GracefulStop`) — reuse these in `cfm.train`
- `src/cfm/smoke/gpu_check.py`, `src/cfm/smoke/resume_test.py`; `tests/test_resume.py`
- `scripts/check_wandb_run.py`, `scripts/wandb_sync.sh` (offline fallback, currently unneeded)
- Eval (M2): `configs/data.yaml` `targets:` section; `src/cfm/data/splits.py` (`split_codes` with embargo, `prediction_points`), `targets.py` (z_h with scale σ_t·sqrt(Σ season), RV_h, 66 bins), `src/cfm/eval/metrics.py` (CE, binned CRPS, randomized PIT, rank IC, QLIKE, log-MAE, `day_bootstrap`), `baselines.py` (marginal, Gaussian, EWMA, log-HAR, GARCH(1,1) with stable h-step sums), `evaluate.py` (`python -m cfm.eval.evaluate baselines`); job `baselines.sh`; tests `test_targets_baselines.py`. Fitted params: `data/stats/baselines.json`.
- Backbone (M3 + v2): `configs/backbone.yaml` (**v2 defaults**: d256/L6/H4/SwiGLU 768/dropout 0.1; context 128 patches, warm-up 96; AdamW lr 3e-4, WSD 3,000 steps (warm-up 300, decay last 50%), batch 256, bf16, compile; ret_weights 1/1/0.5/0.25; flip_prob 0.5; eval every 250 on 2,048 val windows; early stop 12; save_points), `configs/backbone_v1.yaml` (v1, frozen). `src/cfm/data/dataset.py` (`build_patch_data`, `train_starts`, `start_weights` (√size), `WindowSampler`, `eval_windows`; test never loaded, asserted), `src/cfm/data/augment.py` (sign flip), `src/cfm/model/blocks.py`, `backbone.py` (patch embed → blocks → 4×66 return logits + 2 vol log-ratios to EWMA; optional input mask `drop_features`), `src/cfm/losses.py` (CE with optional soft targets, z-loss, QLIKE), `src/cfm/eval/model_eval.py` (`GPUData`, `predict_points` (+ `symmetric` CE), `compare`), `src/cfm/train.py` (`prepare` + `train_run`; CLI `--run-dir --set k=v --symbols --total-steps --max-steps --max-minutes --shuffle-train-targets --input-lag --no-compile --name --group`; `best.pt`/`last.pt`, `eval/val_vs_baselines.csv`, `val_summary.json` (incl. best_step), `val_points.npz`). Job `jobs/train.sh` (gpul40s, 2 h, run name = job name: `bsub -J <name> < jobs/train.sh`).
- Analysis (session 3): `src/cfm/eval/diagnostics.py` (vol-scale vs directional CE split, zero returns, reversal ICs; `bsub -J cfm_diag_<run> < jobs/diagnostics.sh`), `jobs/controls.sh` (shuffle / lag-1 control array), `src/cfm/sweep.py` + `configs/sweeps/*.yaml` + `jobs/sweep.sh` (`bsub -J "cfm_<sweep>[1-N]" < jobs/sweep.sh`), `jobs/sweep_smoke.sh` (CPU smoke, `configs/sweeps/smoke.yaml`), `src/cfm/eval/compare_runs.py` (paired CIs; run as a CPU job, ~2–3 GB RAM; always pass `--out` across sweeps). Tests `test_diagnostics.py`, `test_regularisation.py`.
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
- **Seed noise ≈ 0.01 in val loss** (round 3) is larger than `compare_runs`' paired CIs, which only cover evaluation noise; use ≥ 2 seeds for small differences.
- **Exact-zero returns** (10% at h1) lie on the bin edge at 0 and are binned into the upper bin (`side='right'`); anything symmetric around z (soft targets, mirrored distributions) must account for it.
- `compare_runs` writes to the parent of the *first* run dir unless `--out` is given (it overwrote reg1/compare.csv once).
- `cfm.sweep` calls `torch._dynamo.reset()` per run (recompile limit); the data build is shared, so overrides of `data.*` / `eval.stride` are refused.
- Training is deterministic: the same config + seed reproduces a run exactly (reg1/ref == cfm_bb_v1).
- Disk at session end: `data/` 8.9 GB, `.venv` 9.9 GB (du); `data/runs` 472 MB.
- `/dtu/blackhole` and `/work3` need HPC-support-created dirs; the user chose not to request them.

## How to resume work
```bash
cd ~/crypto-fm
bjobs -a                       # 29759200 / 29759201 (v2 + diagnostics) finished?
bsub < jobs/tests.sh           # sanity: unit tests
source jobs/common.sh          # login-node env for light work (wandb API, etc.)
```
