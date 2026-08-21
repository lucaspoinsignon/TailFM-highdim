"""Train the baseline generators (TimeVAE, TimeGAN, Tail-GAN) on your own
return series and evaluate them with the SAME pipeline as fit_returns.py, so
the numbers are directly comparable with tailfm.

Usage (same data conventions as fit_returns.py):
    python run_baselines.py --data returns.csv --n 24 --gen 50000
    python run_baselines.py --data returns_prices.csv --prices --models timevae,tailgan
    python run_baselines.py --data returns.csv --quick            # CPU smoke test

To put tailfm in the same comparison table, first run fit_returns.py (same
--data/--n/--test-frac/--horizon/--seed!) and pass its generated windows:
    python fit_returns.py  --data returns.csv --n 24 --outdir run_out
    python run_baselines.py --data returns.csv --n 24 --tailfm-gen run_out/generated_windows.npy

Pipeline per baseline: temporal train/test split -> min-max/max-abs scaling
(each reference implementation's own convention, applied inside the baseline)
-> training -> sampling -> tail diagnostics (tailfm.evaluate.print_report),
portfolio VaR/CVaR with bootstrap CIs (tailfm.risk.estimate_risk) and a Kupiec
backtest on the held-out period -> comparison table + figures.

Everything printed is also written to {outdir}/report.log (override with --log),
and the figures are one PNG per diagnostic (figures.py): qq_lower_tail.png,
tail_dependence.png, portfolio_loss_survival.png, empirical_distributions.png.

Note: unlike fit_returns.py, NO rank-recalibration is applied to the baselines
-- they are evaluated exactly as generated, which is the point of the
comparison (tailfm's EVT marginal guarantee is part of the model).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csvio import load_returns, feature_names_from_csv
from figures import save_all_figures
from run_logging import tee_output
from tailfm import (estimate_risk, kupiec_test, portfolio_losses, make_windows,
                    print_report)
from baselines import BASELINES


def parse_args():
    ap = argparse.ArgumentParser()
    # ---- data arguments: keep identical to fit_returns.py -------------------
    ap.add_argument("--data", required=True, help="CSV or .npy of shape (T, f)")
    ap.add_argument("--prices", action="store_true",
                    help="input is prices, not returns")
    ap.add_argument("--n", type=int, default=24, help="window length")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--gen", type=int, default=50_000, help="# generated windows")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--weights", type=str, default=None,
                    help="comma-separated portfolio weights (default: equal)")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="baseline_out")
    ap.add_argument("--log", type=str, default=None,
                    help="text file receiving a copy of everything printed "
                         "(default: {outdir}/report.log)")
    # ---- model selection ----------------------------------------------------
    ap.add_argument("--models", type=str, default="timevae,timegan,tailgan",
                    help="comma-separated subset of: timevae,timegan,tailgan")
    ap.add_argument("--tailfm-gen", type=str, default=None,
                    help="path to generated_windows.npy from a fit_returns.py "
                         "run with the same data settings, to include tailfm "
                         "in the comparison table")
    ap.add_argument("--quick", action="store_true",
                    help="tiny training budgets for a CPU smoke test")
    ap.add_argument("--reuse", action="store_true",
                    help="for each requested model, load {outdir}/gen_<model>.npy "
                         "if it exists instead of retraining (recover from a "
                         "crashed evaluation, or re-run the evaluation and "
                         "figures without paying for training again)")
    # ---- per-model budgets (reference defaults; see baselines/*.py) ---------
    ap.add_argument("--timevae-epochs", type=int, default=1000)
    ap.add_argument("--timevae-batch", type=int, default=16)
    ap.add_argument("--timevae-recon-wt", type=float, default=3.0,
                    help="reference default 3.0; raise (e.g. 100) to counteract "
                         "posterior collapse on near-i.i.d. return windows")
    ap.add_argument("--timevae-latent", type=int, default=8)
    ap.add_argument("--timegan-iters", type=int, default=10_000,
                    help="iterations PER PHASE (reference default 50000)")
    ap.add_argument("--timegan-batch", type=int, default=128)
    ap.add_argument("--tailgan-epochs", type=int, default=3000)
    ap.add_argument("--tailgan-batch", type=int, default=1000)
    ap.add_argument("--tailgan-lr-g", type=float, default=1e-6)
    ap.add_argument("--tailgan-lr-d", type=float, default=1e-7)
    ap.add_argument("--tailgan-alphas", type=str, default="0.05",
                    help="comma-separated PnL tail levels the score targets "
                         "(reference default 0.05). To align training with the "
                         "evaluation levels use 0.05,0.01,0.005; note the "
                         "alpha=0.005 tail has only ~batch_size*0.005 order "
                         "statistics per batch, so keep the batch large.")
    args = ap.parse_args()

    if args.quick:
        args.timevae_epochs = 30
        args.timegan_iters = 100
        args.tailgan_epochs = 30
        args.tailgan_batch = 128
        args.gen = min(args.gen, 2048)

    # Fail fast: validate all paths BEFORE spending compute on training.
    if args.tailfm_gen and not os.path.exists(args.tailfm_gen):
        ap.error(f"--tailfm-gen file not found: {args.tailfm_gen}\n"
                 "(checked up front so no training time is wasted; run "
                 "fit_returns.py first or fix the path)")
    if not os.path.exists(args.data):
        ap.error(f"--data file not found: {args.data}")
    return args


def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    alphas = (0.95, 0.99, 0.995)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in BASELINES]
    assert not unknown, f"unknown model(s) {unknown}; choose from {list(BASELINES)}"

    # --------------------------------------------------------------- data
    r = load_returns(args.data, args.prices)
    T, f = r.shape
    names = feature_names_from_csv(args.data, f)
    split = int((1.0 - args.test_frac) * T)
    train_r, test_r = r[:split], r[split:]
    real = make_windows(train_r, args.n, args.stride)
    print(f"data: T={T}, f={f} ({', '.join(names)}) | "
          f"train windows {real.shape} | test T={test_r.shape[0]}")

    w = (np.array([float(v) for v in args.weights.split(",")])
         if args.weights else np.full(f, 1.0 / f))
    assert w.size == f, "--weights length must equal the number of features"
    L_test = portfolio_losses(make_windows(test_r, args.horizon,
                                           stride=args.horizon),
                              weights=w, horizon=args.horizon)

    hparams = {
        "timevae": dict(max_epochs=args.timevae_epochs,
                        batch_size=args.timevae_batch,
                        reconstruction_wt=args.timevae_recon_wt,
                        latent_dim=args.timevae_latent),
        "timegan": dict(iterations=args.timegan_iters,
                        batch_size=args.timegan_batch),
        "tailgan": dict(n_epochs=args.tailgan_epochs,
                        batch_size=args.tailgan_batch,
                        lr_G=args.tailgan_lr_g, lr_D=args.tailgan_lr_d,
                        alphas=tuple(float(a) for a in
                                     args.tailgan_alphas.split(","))),
    }

    # ------------------------------------------------ train, generate, score
    gens: dict[str, np.ndarray] = {}
    for name in models:
        cache = f"{args.outdir}/gen_{name}.npy"
        if args.reuse and os.path.exists(cache):
            gen = np.load(cache)
            assert gen.shape[1:] == (args.n, f), \
                f"cached {cache} has window shape {gen.shape[1:]}, expected " \
                f"({args.n}, {f}); delete it or drop --reuse to retrain"
            print(f"\n################ {name} (reusing {cache}, "
                  f"{gen.shape[0]} windows) ################")
        else:
            print(f"\n################ {name} ################")
            gen = BASELINES[name](real, args.gen, seed=args.seed,
                                  device=args.device, **hparams[name])
            np.save(cache, gen)
        gens[name] = gen

    if args.tailfm_gen:
        gens["tailfm"] = np.load(args.tailfm_gen)
        assert gens["tailfm"].shape[1:] == (args.n, f), \
            "--tailfm-gen windows have incompatible shape; rerun fit_returns.py " \
            "with the same --n and --data"

    summary: dict[str, dict] = {}
    for name, gen in gens.items():
        print(f"\n=== Diagnostics: {name} (vs real train windows) ===")
        print_report(real, gen, feature_names=names)
        report = estimate_risk(gen, alphas=alphas, weights=w,
                               horizon=args.horizon, n_boot=200, seed=args.seed)
        summary[name] = {}
        print(f"\n=== {name}: portfolio risk (h={args.horizon}) and Kupiec "
              f"backtest (held-out N={L_test.size}) ===")
        for a in alphas:
            rp = report[a]
            k = kupiec_test(L_test, rp["var_gpd"], a)
            summary[name][a] = (rp["var_gpd"], rp["cvar_gpd"],
                                k["exceedances"], k["expected"], k["p_value"])
            print(f"a={a:5.3f}: VaR {rp['var_gpd']:.5f} "
                  f"[{rp['var_ci'][0]:.5f},{rp['var_ci'][1]:.5f}]  "
                  f"CVaR {rp['cvar_gpd']:.5f} "
                  f"[{rp['cvar_ci'][0]:.5f},{rp['cvar_ci'][1]:.5f}]  | "
                  f"exceed {k['exceedances']}/{k['expected']:.1f}  "
                  f"p={k['p_value']:.3f}")

    # ------------------------------------------------------ comparison table
    print("\n" + "=" * 78)
    print("MODEL COMPARISON -- portfolio VaR/CVaR (GPD-refined) and Kupiec "
          "p-value on held-out data")
    print("=" * 78)
    header = f"{'model':>10s}" + "".join(
        f" | a={a:.3f}: VaR    CVaR    exc    p " for a in alphas)
    print(header)
    for name, per_a in summary.items():
        row = f"{name:>10s}"
        for a in alphas:
            v, c, exc, expd, p = per_a[a]
            row += f" | {v:7.4f} {c:7.4f} {exc:3d}/{expd:4.1f} {p:5.3f}"
        print(row)
    print("(Kupiec: p > 0.05 means the VaR level is not rejected; exc/exp = "
          "observed vs expected exceedances)")

    # ----------------------------------------------------------------- figures
    # One PNG per diagnostic (all models overlaid inside each), see figures.py.
    paths = save_all_figures(real, gens, names, args.outdir, weights=w,
                             horizon=args.horizon)
    print("\nSaved: " + f"{args.outdir}/gen_<model>.npy, "
          + ", ".join(os.path.basename(p) for p in paths))


def main():
    args = parse_args()
    log_path = args.log or f"{args.outdir}/report.log"
    with tee_output(log_path, header="run_baselines.py"):
        run(args)
    print(f"Terminal report saved to {log_path}")


if __name__ == "__main__":
    main()
