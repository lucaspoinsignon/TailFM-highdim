# TailFM

Tail-aware flow matching for multivariate return generation, targeted at VaR / CVaR.

EVT marginals (empirical body + GPD tails) map each feature to a $t_\nu$ latent space;
a conditional flow-matching model with a Student-$t$ base transports the copula there;
the inverse PIT maps back. Steps 1–4 need only numpy/pandas/scipy/matplotlib, steps 5+
also need torch.

```bash
pip install numpy pandas scipy matplotlib torch
```

## 1. Raw extract → one price series per VALOR

Input: semicolon-separated `VALOR;FI_ID;PRICE_TYPE;PRICE_DATE;PRICE;CURRENCY`. A VALOR
appearing under several PRICE_TYPEs or CURRENCYs is collapsed to one series.

```bash
python 01_panel.py --raw data/raw.csv --out data/prices.csv --currency USD
```

Read the printed currency mix. Mixed currencies mean the columns are in different units
and their log returns differ by an FX return, which contaminates every cross-sectional
statistic downstream.

## 2. Prices → log returns

Drops series that break the EVT stage, stale quotes, and near-duplicate columns.

```bash
python 02_returns.py --data data/prices.csv --out data/returns.csv
python check.py --data data/returns.csv
```

`check.py` runs the GPD fits for real on the training rows and fails if any feature
transforms to `|z| > 100`. Note it cannot be run on `data/prices.csv`: `01_panel.py`
emits a ragged panel and only `02_returns.py` handles the missingness.

## 3. Plot and explore

```bash
python 03_plot.py --data data/returns.csv --out fig/returns.png
python 04_analyse.py --data data/returns.csv --out fig/dependence.png
python evt_ks.py --data data/returns.csv --out fig/ks.png
```

`04_analyse.py` reports the effective rank of the rank-transformed panel — with $f$
columns of effective rank $k$, the flow is asked to model an $f$-dimensional law
supported near a $k$-dimensional set. `evt_ks.py` is the only out-of-sample check on
the marginals.

## 4. Remove problem series

```bash
python drop_valors.py --data data/returns.csv --out data/returns_clean.csv \
    --valors 4155686,4155690,4157124
```

A single bad price print produces a near-antisymmetric pair of log returns
($r_t + r_{t+1} \approx 0$); mask the cell in the price panel rather than dropping the
instrument. A move that does not reverse is a corporate action and needs the column
dropped.

## 5. Fit

```bash
python fit_returns.py --data data/returns_clean.csv --outdir runs/final
```

The defaults are the settled configuration: `--nu 5 --q-tail 0.05 --pos-std 0.1
--n 24 --test-frac 0.2 --horizon 10 --steps 20000 --gen 20000 --d-model 512
--ode-steps 100`. Writes `report.log`, `generated_windows.npy`, `model_ema.pt`,
`marginals.pkl` and four diagnostic PNGs into `--outdir`. `--no-figures` drops the
PNGs and `--no-report` the printed tables; neither is read downstream, so both are
worth dropping in a sweep.

Three defaults are not arbitrary:

- **`--nu 5`.** The PIT makes every marginal exactly $t_\nu$ for any $\nu$, so this is
  a design parameter, not an estimate. $E z^4 = \infty$ for $\nu \le 4$, which gives
  the CFM target infinite-variance gradients; $\nu = 5$ is the smallest value without
  that. Across $\nu \in \{3, 5, 8\}$ the base's donated tail dependence varies 7.7×
  while the generated copula moves by 2%, so the choice is free on that axis.
- **`--pos-std 0.1`.** `forward()` computes `in_proj(x) + pos`, and the usual
  `std=0.02` makes position ~3% of the token signal. At that scale the model stays
  time-exchangeable and produces zero within-window volatility clustering — generated
  windows are statistically indistinguishable from real windows with the time axis
  permuted. `0.1` reproduces the observed squared-return ACF; larger values overshoot
  it and start reproducing individual training windows.
- **pooled $\xi$ with a floor at 0** (`evt_shrink.py`, on by default). At ~60
  exceedances per tail, $\mathrm{se}(\hat\xi) \approx 0.17$ against a cross-sectional
  sd of ~0.20, so most apparent heterogeneity is noise. The floor removes the finite
  GPD endpoint that would otherwise cap generated losses at the worst one already
  observed.

## 6. Evaluate

```bash
python summarize_runs.py --data data/returns_clean.csv runs/final
python novelty.py --data data/returns_clean.csv --runs runs/final
python 06_compare.py --data data/returns_clean.csv --run runs/final --out fig/windows.png
```

`novelty.py` is the one that cannot be passed by memorisation; the others can.

## 7. Baselines

```bash
python run_baselines.py --data data/returns_clean.csv \
    --tailfm-gen runs/final/generated_windows.npy --outdir runs/baselines
```

## Files

| | |
|---|---|
| `01_panel.py` … `06_compare.py` | data pipeline and plots |
| `check.py` | validates a returns CSV; run after step 2 |
| `csvio.py` | shared CSV loader |
| `fit_returns.py` | model runner |
| `tailfm/` | the model: `evt`, `base`, `model`, `cfm`, `risk`, `evaluate`, `data` |
| `evt_shrink.py` | pooled GPD tail estimation, applied after `MarginalEnsemble.fit` |
| `evt_ks.py` | out-of-sample PIT goodness-of-fit per feature |
| `figures.py` | diagnostic PNGs, shared by steps 5 and 7 |
| `summarize_runs.py`, `novelty.py` | cross-run evaluation |
| `run_baselines.py`, `baselines/` | TimeVAE, TimeGAN, Tail-GAN |
| `run_logging.py` | tees stdout to `report.log` |

## Known limitations

- The marginals are unconditional, so they are specific to the calibration period. On
  our data the held-out window is ~25% calmer, which shows up as ~0.43 of the promised
  exceedances *and* as $\lambda$ falling 0.288 → 0.212 — one regime shift in both. The
  fix is conditional EVT (McNeil–Frey), not a better GPD fit.
- The Kupiec backtest has no power here: $T_{\text{test}}/h \approx 30$ blocks gives
  0.3 expected exceedances at $\alpha = 0.99$.
- With ~50 independent windows behind 27495 pairwise targets, the copula is close to
  the identifiability limit; `err/flr` in `summarize_runs.py` and the novelty ratio
  should be read together.
