# Crypto OHLCV Foundation Model — Implementation Plan

## 0. Goal and guiding principles

Pretrain a small causal transformer on 1-minute OHLCV candles from **Binance USD-M (USDT-margined) perpetual futures** (target ≥ ~100M candles). Then use it as a backbone for a position-sizing network that outputs a **position in [−1, 1]: −1 = full short, 0 = flat, +1 = full long**.

**The data has not been downloaded yet.** The first goal of the project is to acquire, verify and store the raw 1-minute futures data (Phase 0) before any modeling work starts.

- **Compute environment:** the **entire project runs on the DTU HPC cluster**. All work (download, preprocessing, training, evaluation, backtests) is submitted as **bash job scripts to the LSF scheduler** (`bsub < jobs/<script>.sh`). See section 1.1.
- **Hardware:** single NVIDIA A100 40GB (`gpua100` queue). Compute is *not* the bottleneck. A full training run of a ~5–30M-parameter model takes minutes to an hour. Spend compute on ablations and evaluation, not model size.
- **Main constraint:** low signal-to-noise. Next-minute direction is barely predictable; volatility is quite predictable. Expect small models to win and scaling to saturate early.
- **Non-negotiables:**
  1. **Strict causality.** Every feature, normalization statistic and target scale at time `t` uses only data up to `t`.
  2. **Distributional targets** (binned cross-entropy, QLIKE). No MSE on price levels.
  3. **Time-based splits** with embargo. Never random splits.
  4. **Beat simple baselines** (marginal distribution, EWMA/HAR/GARCH) before trusting anything.
  5. **The test set is evaluated once**, at the very end.
  6. **Every P&L number includes trading costs and funding payments.**
- **Scope:** research and backtesting only. Live trading and order execution are out of scope.

---

## 1. Repository layout and compute environment

```
crypto-fm/
  jobs/              # LSF bash job scripts, one per pipeline step (see 1.1)
    common.sh        # shared setup: modules, venv activation, paths, logging
    download.sh  convert.sh  clean.sh  features.sh
    train.sh  ladder.sh  eval.sh  backtest.sh  tests.sh
  env/
    setup_env.sh     # one-time: load modules, create venv, install pinned requirements
    requirements.txt
  logs/              # LSF stdout/stderr (%J_%I.out/.err)
  configs/
    data.yaml  universe.yaml  backbone.yaml  tokenizer.yaml  policy.yaml
  data/
    raw/
      zips/                           # downloaded monthly/daily zip files + checksums
      parquet/                        # one raw parquet per symbol (klines)
      funding/                        # one parquet per symbol (funding rates)
      manifest.csv                    # download + verification log
    processed/  stats/                # stats = train-only fitted statistics
  src/cfm/
    data/
      universe.py    # list symbols, filter, rank by liquidity
      download.py    # parallel, resumable downloader with checksum verification
      convert.py     # zip/CSV -> parquet, timestamp normalization
      validate.py    # coverage, gaps, OHLC validity, REST spot-checks
      load.py        # read raw klines + funding
      clean.py       # gaps, jumps, illiquid segments
      vol.py         # per-candle variance, EWMA, seasonality, floors
      features.py    # candle + context features
      splits.py      # walk-forward splits + embargo
      dataset.py     # windowed PyTorch dataset, patching, sampling
    model/
      blocks.py      # RMSNorm, RoPE, attention (QK-norm), SwiGLU
      backbone.py    # patch embedding + transformer
      heads.py       # binned-return heads, volatility head
    tokenizer/       # Phase 6 (optional)
      dvae.py  train_tokenizer.py
    losses.py        # CE over bins, QLIKE, z-loss
    train.py         # pretraining loop
    eval/
      baselines.py   # marginal, EWMA, HAR-RV, GARCH
      metrics.py     # CE gain, rank IC, CRPS, calibration, QLIKE
      evaluate.py
    policy/
      position_head.py  train_policy.py  backtest.py
  scripts/           # CLI entry points (download, prepare_data, train, ladder, eval, backtest)
  tests/             # incl. causality tests
```

**Stack:** Python 3.11+, PyTorch 2.x (bf16 autocast, `torch.compile`, SDPA/FlashAttention), polars or pandas, pyarrow/parquet, numpy, `arch` (GARCH), pytest, and **Weights & Biases (`wandb`)** for experiment monitoring. Bash for all job scripts.

### 1.1 Compute environment: DTU HPC (LSF)

**Scheduler.** DTU HPC uses IBM LSF 10. Every pipeline step is a bash job script with `#BSUB` directives, submitted with `bsub < jobs/<script>.sh`. Monitor with `bjobs`, inspect running output with `bpeek <jobid>`, cancel with `bkill <jobid>`.

**Queues.**
- **GPU training and evaluation:** `gpua100` (A100 PCIe; nodes with 40 GB and 80 GB exist). Design for 40 GB, so the code never depends on landing on an 80 GB node.
- **Alternatives:** `gpul40s` (L40S, 48 GB) supports bf16 and works for the same code. `gpuv100` (V100, 16/32 GB) has **no bf16**, so make precision configurable: bf16 on A100/L40S, fp16 + `GradScaler` on V100.
- **CPU-only steps** (download, convert, clean, features, GARCH/HAR baselines, tests): the cluster's CPU batch queue (e.g. `hpc`; confirm the name in the DTU HPC docs). Don't occupy GPU nodes with CPU work.
- **Interactive debugging:** the interactive GPU nodes (e.g. `a100sh`) for short tests only; never for long runs.

**Limits and resulting design rules.**
- **Maximum walltime on GPU queues is 24 hours.** All training must **checkpoint regularly** (model, optimizer, scheduler, data-sampler state, RNG states, step) and support `--resume`. Long jobs are chained with dependencies: `bsub -w "done(<jobname>)" < jobs/train.sh`.
- Request ~4 CPU cores per GPU (the DTU examples do this) and use them as DataLoader workers.
- Use `mode=exclusive_process` for GPUs.

**Template GPU job script (`jobs/train.sh`):**
```bash
#!/bin/bash
#BSUB -q gpua100
#BSUB -J cfm_train
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 23:30
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

set -euo pipefail
source jobs/common.sh            # modules, venv, PROJECT/DATA paths, prints node + GPU info
python -m cfm.train --config configs/backbone.yaml --resume auto \
       --run-dir "$RUNS_DIR/$LSB_JOBNAME" --job-id "$LSB_JOBID"
```
Leave margin below the 24 h limit (e.g. 23:30) and also save a checkpoint when the remaining walltime drops below ~15 minutes.

**Job arrays for parallel CPU work and sweeps.** Use LSF job arrays, e.g. `#BSUB -J "convert[1-80]"`, where each array element processes one symbol via `$LSB_JOBINDEX`. Do the same for the scaling ladder (one array element per model size × data fraction) and learning-rate sweeps. Name logs `logs/%J_%I.out`.

**Environment (`env/setup_env.sh`, run once).** Load the Python and CUDA modules (check `module avail` for current versions), create a venv in the project directory, and install pinned requirements, including the PyTorch wheel matching the loaded CUDA version. `jobs/common.sh` loads the same modules and activates the venv in every job, so all jobs run in an identical environment.

**Storage.**
- Home directories have a small quota. Put raw zips, parquet, processed data, checkpoints and runs on the cluster's scratch/work filesystem (check the DTU HPC docs for the path and quota, e.g. a `/work3/$USER`-style location). Define it once as `DATA_ROOT` in `jobs/common.sh`.
- Optionally copy the processed dataset to node-local scratch at job start for faster I/O.

**Network.**
- Check whether compute nodes have outbound internet access. If they don't, run the Phase 0 downloads on the login or transfer node (light, I/O-bound work; follow the cluster's usage rules for those nodes), and run everything else as batch jobs.
- W&B needs outbound internet from the compute node; see 1.2 for the offline fallback.

**Reproducibility.** Every job writes the resolved config, git commit hash, LSF job ID, node name, GPU model, and library versions into its run directory, and logs the same to W&B (1.2).

**Working with Claude Code on the cluster.**
- Claude Code runs on the login node (inside `tmux`, so sessions survive SSH disconnects) or in an interactive session. It edits code, runs quick tests and **submits all real work with `bsub`**. It never runs training, preprocessing or other heavy compute directly on the login node.
- After submitting, check progress with `bjobs`, `bpeek` and the W&B run, rather than blocking on long jobs.
- The home directory is on network storage shared across nodes, so use Claude Code from one node at a time.

### 1.2 Experiment monitoring: Weights & Biases

**Setup (once):** `pip install wandb` in the venv, then `wandb login` on the login node. The API key is stored in `~/.netrc`. **Never commit the key** or put it in job scripts or configs.

**In `jobs/common.sh`:**
```bash
export WANDB_PROJECT=crypto-fm
export WANDB_ENTITY=<user-or-team>        # to be filled in
export WANDB_DIR="$DATA_ROOT/wandb"        # keep W&B files off the small home quota
export WANDB_MODE=${WANDB_MODE:-online}    # set to "offline" if compute nodes lack internet
```

**Offline fallback:** if compute nodes have no internet access, jobs run with `WANDB_MODE=offline`, and a small script on the login node runs `wandb sync "$WANDB_DIR"/wandb/offline-run-*` periodically (or after each job) to upload the runs.

**Runs across chained 24-hour jobs:** each training run gets a W&B run ID generated on its first job and saved in the run directory. Resumed jobs call `wandb.init(id=<saved id>, resume="allow")`, so one training run shows as one continuous W&B run across job boundaries.

**Organization:**
- `wandb.config` holds the full resolved config, plus the git hash, LSF job ID, node and GPU model.
- Use `group` for related runs (e.g. `ladder`, `lr-sweep`, `policy-v1`) and tags for the phase and model size.
- Scaling-ladder and learning-rate sweeps (LSF job arrays) log each array element as its own run in the same group.

**What to log:**
- **Training (every ~50 steps):** total loss, each CE head, QLIKE, z-loss, learning rate, gradient norm, tokens/s, GPU memory.
- **Validation (every ~500 steps):** the same losses, CE gain over the marginal baseline, rank IC per horizon, and QLIKE vs. EWMA/HAR.
- **Tokenizer (Phase 6):** reconstruction loss and per-row codebook perplexity.
- **Policy (Phase 10):** Sharpe, P&L split into gross / costs / funding, turnover, average |pos|, and equity curves as W&B plots.
- **Artifacts:** the best checkpoint per run and evaluation tables as W&B artifacts (optional; checkpoints always also stay on the work filesystem).

**Alerts (optional):** `wandb.alert` on NaN loss, divergence (loss spike above a threshold) or job completion.

---

## 2. Phase 0 — Data acquisition (first goal)

Download raw 1-minute klines and funding rates for a broad universe of Binance USD-M perpetual futures. Store them unnormalized: every transform in later phases needs raw prices. (The rolling min-max representation the user originally planned is computed later, causally, as an extra input feature in Phase 2.)

### 2.1 Source
- **Binance public data** (https://data.binance.vision, repo: https://github.com/binance/binance-public-data). It's free, needs no API key, and serves zipped CSVs per symbol: monthly files, plus daily files for the current month.
- **Klines (required):** `https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/1m/{SYMBOL}-1m-{YYYY-MM}.zip`, with a matching `.CHECKSUM` file (SHA-256) next to each zip. Daily files live under `.../futures/um/daily/klines/...`.
- **Kline CSV columns:** open_time, open, high, low, close, volume, close_time, quote_volume, n_trades, taker_buy_base_volume, taker_buy_quote_volume, ignore.
- **Funding rates (required, for P&L and as a feature):** `.../futures/um/monthly/fundingRate/{SYMBOL}/{SYMBOL}-fundingRate-{YYYY-MM}.zip`. Verify the exact path and columns against the bucket listing before writing the parser.
- **Optional extras** (verify availability per symbol and period before relying on them):
  - mark-price, index-price and premium-index klines (`markPriceKlines`, `indexPriceKlines`, `premiumIndexKlines`), giving the basis/premium feature
  - the daily `metrics` files (open interest, long/short ratios), which only cover more recent years
- ⚠️ **Timestamps:** detect the unit per file by magnitude (milliseconds vs. microseconds) rather than assuming. Binance switched *spot* files to microseconds from 2025; futures files may differ, so the parser must handle both. Convert everything to UTC.
- ⚠️ Some files have a header row and some don't. Detect and handle both.
- **History:** the first USDT perpetual (BTCUSDT) launched in late 2019; most altcoin perps start in 2020 or later.

### 2.2 Universe selection (`universe.py`)
1. **List all symbols from the bucket listing,** not just currently traded contracts. This includes delisted perps and reduces survivorship bias. Listing endpoint: `https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?prefix=data/futures/um/monthly/klines/&delimiter=/`, paginated.
2. **Keep only USDT-margined perpetuals (`*USDT`).** Exclude:
   - quarterly delivery contracts (symbols containing `_YYMMDD`)
   - USDC-margined contracts (`*USDC`) and other quote currencies
   - stablecoin-vs-stablecoin contracts
   - index or basket contracts (e.g. `*DOMUSDT`, `DEFIUSDT`, if present)
   - Keep `1000*` contracts (e.g. `1000PEPEUSDT`): the scaled contract size doesn't affect log returns.
3. **Rank by median daily quote volume** (use 1-day klines, which are fast to download). Keep the top ~50–100 symbols with at least 12 months of history.
4. **Size check:** one asset-year is about 525,600 candles. **Target ≥ ~100M candles**, roughly 200 asset-years (e.g. ~60 contracts × ~3.5 years). Adjust the universe until the target is met, and log the final count.
5. Save the chosen universe to `configs/universe.yaml` with the reasoning for each exclusion.

### 2.3 Download (`download.py`)
- Parallel downloads (8–16 workers), with retries and exponential backoff.
- **Resumable:** skip files that already exist and pass checksum verification.
- **Verify every zip** against its `.CHECKSUM` file. Re-download on mismatch; after 3 failures, log the file as failed.
- Write `data/raw/manifest.csv` with one row per file: symbol, data type, period, url, sha256, status, rows, first and last timestamp.
- Months not yet published as monthly files: fetch the daily files instead.
- **On DTU HPC:** wrap the downloader in `jobs/download.sh`. If compute nodes have internet, submit it to the CPU queue; otherwise run it on the login/transfer node (see 1.1). Conversion and validation run as CPU job arrays (`jobs/convert.sh`, one element per symbol).

### 2.4 Convert and validate (`convert.py`, `validate.py`)
- **Klines:** convert to one parquet per symbol with columns `ts (UTC), open, high, low, close, volume, n_trades, taker_buy_base`, sorted, deduplicated. Stored losslessly as integers (prices in ticks, volumes in lots, delta-encoded) and decoded to float64 on load. `quote_volume` and `taker_buy_quote` are dropped to save space (see 15.2).
- **Funding:** one parquet per symbol with `funding_time (UTC), funding_rate` (plus mark price if present). Note the funding interval can differ by symbol and change over time (commonly 8h, sometimes 4h or 1h), so store the actual timestamps rather than assuming a schedule.
- **Validation report per symbol:**
  - rows per month vs. minutes in that month
  - duplicate timestamps and gap statistics
  - OHLC validity (H ≥ max(O, C), L ≤ min(O, C))
  - zero-volume share
  - funding coverage: every funding timestamp within the kline date range is present
- **Spot-check:** compare ~10 random days for a few symbols against the public futures REST endpoint `https://fapi.binance.com/fapi/v1/klines` (no API key needed for market data).

**Acceptance (M0):** manifest complete with all checksums verified (or failures explained); ≥ ~100M candles stored as raw parquet; funding data stored for every symbol; validation report generated; spot-checks match.

---

## 3. Phase 1 — Data cleaning

Input: the raw parquet files from Phase 0.

Steps:
1. Parse timestamps to UTC, sort, and deduplicate per symbol.
2. Handle gaps:
   - **Short gaps (≤ 15 min):** forward-fill price (O = H = L = C = previous close) with V = 0, and set a `filled` flag.
   - **Longer gaps:** split into separate segments.
3. Split segments at abnormal price jumps, e.g. |1-minute log return| > 0.10, unless confirmed real (make this configurable).
4. Split segments at illiquid or stagnant stretches, e.g. > 15 consecutive identical candles or > 45 consecutive zero-volume candles (configurable).
5. Drop each contract's first few days after listing (launch volatility is unrepresentative; configurable, default 3 days).
6. Drop segments shorter than `warmup (1440) + context + horizon`.
7. Output one parquet file per symbol with a `segment_id` column. Log the statistics: candles kept, segments, and what was dropped and why.
8. Document survivorship bias: which delisted contracts are included, and whether any are missing.

**Acceptance:** a cleaning report per symbol; no NaNs; monotonic timestamps within each segment.

---

## 4. Phase 2 — Features (all causal)

### 4.1 Volatility (`vol.py`)

Per-candle variance estimate (range-based, ~5–8× more efficient than squared returns):

```
Garman–Klass:    v_t = 0.5·ln(H/L)² − (2·ln2 − 1)·ln(C/O)²
Parkinson:       v_t = ln(H/L)² / (4·ln2)                      # fallback
Rogers–Satchell: v_t = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)        # optional, drift-robust
```

- **Smoothing:** `σ²_t = λ·σ²_{t−1} + (1−λ)·v_t`, with `λ = 0.5^(1/half_life)` and a default half-life of 60 minutes.
- **Seasonality:** compute a factor per minute-of-week as the median per-candle volatility at that minute divided by the overall median. **Fit on train only.** Divide it out of `v_t` before smoothing, and store the factor. (Funding times show up as seasonal volatility spikes; this handles them.)
- **Floor:** a per-symbol floor at the 5th percentile of `σ` (train only), to prevent division blow-ups in dead periods.
- **Rule:** features at candle `t` use `σ_{t−1}`, the volatility known *before* candle `t`.

### 4.2 Candle features (`features.py`)

```
body  = ln(C / O)            / σ_{t−1}
upper = ln(H / max(O, C))    / σ_{t−1}    # ≥ 0
lower = ln(min(O, C) / L)    / σ_{t−1}    # ≥ 0
ret   = ln(C / C_{t−1})      / σ_{t−1}
vrel  = ln(1+V) − mean_{1440}(ln(1+V))    # trailing window ending at t−1
```

- Squash outliers with `x ← 5·tanh(x/5)`.
- Optional gap feature `ln(O/C_{t−1})/σ` (usually ≈ 0 for crypto).

### 4.3 Context features (inputs only, never targets)

- `ln σ_{t−1}`: the volatility regime level, needed because normalization removes scale.
- Log volume level, z-scored with trailing statistics.
- **Taker-buy imbalance:** `2 · taker_buy_base / volume − 1` (in [−1, 1]; 0 when volume is 0), an order-flow signal.
- Relative trade count: `ln(1 + n_trades)` minus its trailing mean.
- **Funding:** the most recent *settled* funding rate, and the time until the next funding timestamp. Use only rates already published at time `t`.
- Optional: basis/premium from the premium-index or index-price klines, if downloaded.
- Position within the trailing 1440-minute range (the user's min-max feature), computed causally.
- Time encodings: sin/cos of minute-of-day, sin/cos of day-of-week.
- Flags: `filled`.

### 4.4 Patching

- Patch size `P = 15` candles, aligned to the clock (:00, :15, :30, :45).
- Inside a patch, normalize **all 15 candles with the anchor close and `σ` at the patch start**, so intra-patch volatility changes stay visible. (Per-candle features in 4.2 are the simpler default; make the anchoring mode a config option.)

### 4.5 Tests (must exist and pass)

- **Causality test:** randomly perturb all data after time `t`, recompute features, and assert all features at times ≤ `t` are unchanged. Run this for every feature, for `σ`, and for the funding features.
- **Candle validity:** H ≥ max(O, C) and L ≤ min(O, C) after cleaning.

---

## 5. Phase 3 — Splits and dataset

- **Walk-forward time splits:** train ≤ T1, validation in (T1 + embargo, T2], test in (T2 + embargo, end]. Choose the dates after inspecting coverage, e.g. train ≤ 2023-12-31, validation 2024, test 2025 onward.
- **Embargo:** at least 1 day plus the context length, so no window spans two splits.
- **Optional:** hold out a few whole symbols to measure cross-asset generalization.
- **Fit everything on train only:** seasonal factors, `σ` floors, return-bin edges, tokenizer, and any scalers. Save them to `data/stats/`.
- **Sampling:** draw random (symbol, segment, start) windows, with each symbol weighted by roughly √(its size) so BTC/ETH don't dominate. Weights are configurable.
- **Context:** 128 patches = 1,920 minutes, about 32 hours, plus a 1440-minute feature warmup before the first patch.

---

## 6. Phase 4 — Targets and losses

At each patch position (teacher forcing, causal), predict from the information available at the **end** of that patch:

### 6.1 Return heads: binned cross-entropy
- **Targets:** cumulative log return of the last-trade close over horizons h ∈ {1, 5, 15, 60} minutes, divided by `σ` at prediction time.
- **Bins:** 64 uniform bins over [−5, 5] plus 2 tail bins. Quantile-based edges fitted on train are an alternative to test.
- **Heads:** one linear head per horizon, `d → 66` logits, with cross-entropy loss.

### 6.2 Volatility head: QLIKE
- **Target:** realized variance `RV_h = Σ v_t` (range-based) over the next 15 and 60 minutes, deseasonalized.
- **Head:** outputs `log σ̂²_h`.
- **Loss:** `QLIKE = RV/σ̂² − ln(RV/σ̂²) − 1`.

### 6.3 Total loss
```
L = Σ_h CE_h + w_vol · Σ_h QLIKE_h + 1e-4 · z_loss
```
Default `w_vol = 1`; tune it on validation.

**Sanity check:** at initialization, each CE head's loss should be ≈ ln(66) ≈ 4.19.

---

## 7. Phase 5 — Backbone

```yaml
patch_embed: Linear(P * F -> d)       # F = number of per-candle features (+ context)
d_model: 256
n_layers: 6
n_heads: 4                             # head_dim 64, plain MHA (no KV-cache concerns)
mlp: SwiGLU, hidden 768
norm: pre-norm RMSNorm, final RMSNorm before heads
qk_norm: true
pos: RoPE (base 10000)
bias: false
dropout: 0.1                           # noisy, data-limited regime
init: trunc_normal(0, 0.02); residual output projections scaled by 1/sqrt(2*n_layers)
attention: causal, via F.scaled_dot_product_attention
```
This is about 5M parameters. Parameter count ≈ `L·d²·(4 + 3r)` + embeddings and heads, where `r` = MLP hidden size / d.

---

## 8. Phase 6 (optional, only after Phase 5 beats baselines) — Learned discrete tokenizer

A categorical dVAE over 15-candle patches.

- **Encoder:** a small transformer over the 15 candles' features produces `N × M` logits (default N = 8 rows, M = 64 classes, i.e. 48 bits).
- **Quantization:** straight-through one-hot.
  ```python
  probs = logits.softmax(-1)
  hard = F.one_hot(probs.argmax(-1), M).float()
  z = hard + probs - probs.detach()
  emb = torch.einsum("bnm,nmd->bnd", z, codebooks)
  ```
  Mix in 1% uniform probability: `probs = 0.99*probs + 0.01/M`.
- **Decoder:** reconstructs body / upper / lower / ret / vrel for all 15 candles. Wicks are output through softplus, so every reconstructed candle is valid. Loss: Huber.
- **Coarse-to-fine:** randomly keep only the first k ∈ {1, 2, 4, 8} rows (masking the rest) and add their reconstruction losses, so early rows hold the coarse structure.
- **Anti-collapse:** add `+λ₁ · mean per-sample entropy − λ₂ · entropy of the batch-mean distribution` per row. Log per-row codebook perplexity; alert if it falls below M/4.
- **Forecasting model on top:**
  - Input is the encoder's continuous pre-quantization embedding (or the fused code embeddings).
  - Targets are the next patch's N code rows, predicted **sequentially**, with row i conditioned on rows < i via a small causal transformer or cross-attention (as in Kronos).
- **Judge it by forecasting, not reconstruction.** Ablate:
  1. Phase 5 baseline (continuous patch input, binned CE)
  2. This learned tokenizer
  3. Pretrained Kronos (https://github.com/shiyu-coder/Kronos), zero-shot and fine-tuned

---

## 9. Phase 7 — Training setup

```yaml
optimizer: AdamW, betas (0.9, 0.95), eps 1e-8
weight_decay: 0.1 (none on norms, biases, embeddings)
lr_peak: 1e-3        # sweep {3e-4, 1e-3, 3e-3}
schedule: WSD — 1000 warmup steps, constant, linear decay over the last 20%
batch: 256 sequences × 128 patches
grad_clip: 1.0
precision: bf16 autocast; TF32 enabled
compile: torch.compile
eval: every ~500 steps on validation; keep the best checkpoint; early-stop on validation loss
seeds: fixed; log config + git hash with every run
logging: wandb (see 1.2); one W&B run per training run, resumed across chained jobs
```

---

## 10. Phase 8 — Scaling ladder

| Name | d | layers | heads | ~params |
|---|---|---|---|---|
| XS | 128 | 4 | 2 | ~0.6M |
| S | 192 | 6 | 3 | ~2M |
| M | 256 | 8 | 4 | ~6M |
| L | 384 | 10 | 6 | ~15M |
| XL | 512 | 12 | 8 | ~30M |

- Train each size on 25%, 50% and 100% of the training data, with early stopping.
- Plot best validation loss against parameters and against data. Optionally fit `L(N, D) = E + A/N^α + B/D^β`.
- **Choose the smallest model near the plateau.** Expect saturation in the single-digit millions of parameters.

---

## 11. Phase 9 — Evaluation

**Baselines (`baselines.py`):**
- **Returns:** the unconditional train marginal over bins, and a zero-mean Gaussian with EWMA `σ`.
- **Volatility:** EWMA, HAR-RV (15-minute / 4-hour / 1-day components, linear regression), and GARCH(1,1) per symbol.

**Metrics (`metrics.py`):**
- CE improvement over the marginal (nats and %)
- Rank IC: Spearman correlation of predicted mean return vs. realized, per horizon. Report both pooled time-series and cross-sectional (across symbols at each timestamp).
- CRPS and calibration (PIT histogram, reliability)
- Volatility: QLIKE and MAE of log RV vs. EWMA / HAR / GARCH
- Breakdowns by symbol group, volatility regime and time period

**Statistics:** windows overlap and symbols are correlated, so use **block bootstrap by day** for confidence intervals. Don't trust differences inside the CI.

---

## 12. Phase 10 — Position network

### 12.1 Position definition
- The policy outputs `pos ∈ [−1, 1]` per symbol: **−1 = full short, 0 = flat, +1 = full long.**
- `|pos|` is the fraction of the capital allocated to that symbol, held as a futures position with notional `pos × allocated equity` (i.e. **1× leverage at |pos| = 1**). No leverage beyond 1× unless the user changes the config.
- **Decision frequency:** every 15 minutes, at patch boundaries. Hold the position until the next decision. This cuts turnover.
- **Multi-symbol portfolio (default):** each symbol gets an equal share `1/K` of capital among the K symbols tradable at that time; portfolio P&L is the equal-weighted average of per-symbol P&L. Single-symbol evaluation is also reported.

### 12.2 P&L per decision interval (shared by training and backtest)
```
pnl_t = pos_t · r_{t→t+15}                         # log or simple return of the contract
        − c · |pos_t − pos_{t−1}|                  # trading costs on turnover
        − pos_t · Σ funding_rate_k                 # funding settled inside (t, t+15]
```
- Longs pay funding when the rate is positive and receive it when negative; shorts the opposite. Hence the `−pos_t · funding` term.
- **Cost `c`:** configurable, default = taker fee 0.05% + slippage 0.02% per unit turnover. Check the current Binance USD-M fee schedule and adjust; also run a sensitivity sweep (e.g. c ∈ {0.03%, 0.07%, 0.12%}).

### 12.3 Policies to compare
1. **Baseline rule:** `pos = clip(k · μ̂ / σ̂², −1, 1)`, with μ̂ and σ̂ from the backbone heads and `k` tuned on validation.
2. **Learned head:** frozen backbone → last hidden state (plus μ̂ and σ̂ from the heads, and the current position) → MLP → **tanh** → `pos ∈ [−1, 1]`.
   - Loss on sequences of decisions: `−mean(pnl) / std(pnl)` (Sharpe), or `−mean(pnl) + λ·var(pnl)`.
   - Feed the previous position as input so the head can learn to avoid unnecessary turnover.
   - Stage 2: unfreeze the backbone with a 10× lower learning rate.
3. **Later, optional:** a diffusion scenario head that samples N future paths, with the position chosen by maximizing `E[PnL] − λ·risk − costs` across scenarios, then clipped to [−1, 1].

### 12.4 Backtest (`backtest.py`)
- Event-driven, no lookahead, using exactly the P&L definition in 12.2 (costs and funding included).
- **Metrics:** Sharpe, annualized return, max drawdown, turnover, hit rate, average |pos|, long/short share, and P&L split into gross / costs / funding.
- **Compare against:** flat (always 0), always long (+1), always short (−1), and the baseline rule.
- Tune everything on validation; run the test period **once**.

---

## 13. Milestones and acceptance criteria

| # | Milestone | Done when |
|---|---|---|
| M−1 | HPC environment ready | `env/setup_env.sh` builds the venv; `jobs/common.sh` works; a test GPU job on `gpua100` prints the GPU and runs a tiny bf16 forward/backward; checkpoint-and-resume verified across two chained jobs, appearing as **one continuous W&B run** (online, or offline + synced) |
| M0 | Data acquired | Universe chosen; manifest complete with checksums verified; ≥ ~100M candles + funding stored as raw parquet on the work filesystem; validation report and REST spot-checks pass |
| M1 | Data cleaned, features built | Cleaning report exists; causality and candle-validity tests pass (run via `jobs/tests.sh`) |
| M2 | Baselines | Marginal / EWMA / HAR / GARCH metrics on validation logged |
| M3 | Backbone v1 | Initial CE ≈ ln(66); beats the marginal CE and EWMA QLIKE on validation with CI excluding 0 |
| M4 | Scaling ladder | Plots produced; model size chosen |
| M5 | Volatility | Volatility head beats HAR-RV on validation QLIKE |
| M6 | Tokenizer ablation (optional) | Comparison table: baseline vs. dVAE vs. Kronos |
| M7 | Position network | Policy in [−1, 1] beats flat, always-long and always-short on validation **after costs and funding**; single test run reported |

---

## 14. Pitfall checklist (review before every experiment)

- [ ] No feature, scale or target uses data from after its timestamp (the causality test passes)
- [ ] Target scaling uses `σ` at prediction time, never the target window's own statistics
- [ ] Funding features use only rates already settled at time `t`
- [ ] All fitted statistics (seasonality, floors, bins, tokenizer) come from train only
- [ ] Time-based splits with embargo; no random splits
- [ ] Block-bootstrap confidence intervals; the effective sample size is far below the raw candle count
- [ ] Test set untouched until the final evaluation
- [ ] Trading costs **and funding** included in every P&L number
- [ ] Positions always within [−1, 1]
- [ ] Delisted contracts included where available; survivorship bias documented

---

## 15. Open questions — ask the user before starting

1. Universe size and date range: is "top ~50–100 USDT perps by liquidity, from 2020 onward" acceptable?
2. Fee tier and slippage assumptions for the backtest (default: 0.05% taker + 0.02% slippage)?
3. Portfolio setup: equal-weight across all tradable symbols (default), or one model/position per symbol evaluated separately?
4. W&B entity (user or team name) and project name (default `crypto-fm`)?
5. Is a fixed decision interval of 15 minutes acceptable for the policy?
6. DTU HPC specifics:
   - Which storage location and quota should hold the data and checkpoints (the work/scratch path)?
   - Do compute nodes have outbound internet, or must downloads run on the login/transfer node?
   - Which CPU queue should CPU-only jobs use, and should LSF send email notifications (`#BSUB -B -N`)?

### 15.1 Answers (2026-10-19)

1. Universe: top ~50–100 USDT perps by liquidity, from 2020 onward, as planned.
2. Costs: 0.05% taker + 0.02% slippage per unit turnover (c = 0.07%), plus the sensitivity sweep.
3. Portfolio: equal-weight across tradable symbols (default); single-symbol results also reported.
4. W&B: the logged-in account's default entity (`WANDB_ENTITY` left unset), project `crypto-fm`.
5. Decision interval: fixed 15 minutes.
6. DTU HPC:
   - **Storage: the 30 GB home quota only** (no /work3 or blackhole). `DATA_ROOT = <project>/data`; the venv is `<project>/.venv`. Space is the binding constraint: delete zips and intermediates once verified, avoid caches in home, and monitor with `getquota_zhome.sh`.
   - Internet on compute nodes: tested in M−1 (`jobs/check_net.sh`, `jobs/test_gpu.sh`).
   - CPU queue: `hpc`; no LSF email notifications.
   - Consequence: torch is the CUDA 13 build (smaller wheel), so `gpuv100` (no sm_70 support) is not a fallback; use `gpua100`, `gpul40s` or `gpuh100`.

### 15.2 Data budget decisions (2026-10-19)

- **Maximize candles** within the quota while leaving headroom for checkpoints and other tasks; ≥100M is a minimum, not the target.
- Universe: USDT perpetuals with a COIN underlying (exchangeInfo; TradFi, index, stable and gold contracts excluded), ≥365 days of history, **median daily quote volume ≥ $5M** → 347 symbols, ~647M estimated candles.
- Raw klines stored as lossless integers **without `quote_volume` / `taker_buy_quote`** (~15 B/candle vs ~45 B for float64 parquet). Exact daily quote volume stays available in `data/raw/universe/klines_1d.parquet`; per-minute dollar volume ≈ volume × price.
- Budget: ~10.5 GB for raw klines, ~8.5 GB reserved (checkpoints 3, W&B/logs/eval 1, stats/indices 1, transient 1.5, safety 2).
- **Phases 1–3 must not store a second full copy** of the data: cleaning produces a segment index (plus flags), and features are computed in RAM when the dataset is built (or stored compactly if that proves too slow).

### 15.3 Splits and Phase 1–2 implementation decisions (2026-10-19)

- **Splits (user choice):** train ≤ 2024-12-31 (363.5M cleaned minutes), val 2025-01-01 … 2025-09-30 (115.3M), test 2025-10-01 … 2026-09-30 (159.8M); 3-day embargo at the start of val and test. 296 of 371 symbols have train data. All in `configs/data.yaml`.
- **Cleaning** additionally re-applies the listing warm-up after any gap ≥ 7 days (relistings) and clips `taker_buy_base` to `volume`.
- **Seasonality** is pooled over all symbols' train data (each symbol normalised by its own train mean variance, winsorised at the 99.9th percentile) and uses means, not medians: per-candle variance medians are 0 for illiquid symbols, and 75 symbols have no train data.
- **σ floor** is causal and adaptive, `σ ≥ max(0.2 · slow-EWMA σ (half-life 7 days), 1e-6)`, instead of a train-only 5th percentile, which does not exist for symbols listed after the train period.
- **Normaliser** for candle t is `σ_{t−1} · sqrt(season_t)` (deseasonalised EWMA re-seasonalised with the calendar factor known in advance); use the same scale for return targets in Phase 4.
- Patch-anchored normalisation (§4.4) is left to the Phase 3 dataset.

---

## 16. Pre-M4: regularisation of the return heads (added 2026-10-19, session 3)

**Problem (M3 run `cfm_bb_v1`):** the best validation loss came at step 1,000 (still in warm-up); afterwards train CE keeps falling while validation CE rises, most for long horizons (h60 gain 0.10 → −0.05). The vol heads do not overfit, but the overfitting return heads force an early checkpoint and hurt the shared backbone (with shuffled return targets the vol head beats HAR by 0.041 / 0.024 instead of 0.033 / 0.017). Suspected mechanism: memorisation of market-wide moves (one time point's outcome is shared by ~300 correlated symbols and by 4 overlapping h60 targets; timestamps are identifiable from time encodings and context).

**Infrastructure:** `cfm.train --set key=value` overrides; `loss.ret_weights` (per return horizon), `loss.hl_gauss_sigma` (Gaussian label smoothing over the bins, centred on the hard bin), `model.drop_features`; `cfm.sweep` trains several configurations on one data build (`configs/sweeps/*.yaml`, `jobs/sweep.sh`); `eval.save_points` + `cfm.eval.compare_runs` give paired day-bootstrap CIs between runs on identical validation points.

**Round 1 (`configs/sweeps/reg1.yaml`), one change at a time,** eval every 250 steps, early stop after 12 evals: reference; dropout 0.2 / 0.3; weight decay 0.3; LR 3e-4; HL-Gauss σ = 0.75 bin widths; return weights 1/1/0.5/0.25; no calendar features (mechanism probe); short schedule (4,000 steps, warm-up 500, decay over the last 20%).

**Round 2:** combine the 2–3 helpful changes; repeat the winner with a second seed; one full run `cfm_bb_v2` with the usual final evaluation.

**Selection:** primary = validation loss (Σ CE + Σ QLIKE) of each run's best checkpoint on all validation points, paired CI vs the reference; secondary = step of the best checkpoint, h60 CE degradation after it, QLIKE vs HAR, train–val CE gap. Within CIs prefer the simpler config.

**Deferred:** market-relative (cross-sectionally de-meaned) return targets — attack the mechanism directly but change the task (market-neutral policy in M7); only if round 1 does not fix the overfitting.

**Afterwards (before M4):** stronger yardsticks — HAR-rescaled marginal as the return baseline and reversal signals as IC baselines.
