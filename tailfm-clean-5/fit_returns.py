"""Fit tail-aware flow matching on a multivariate return series.

    python fit_returns.py --data data/returns_clean.csv --outdir runs/final

Input: a CSV (one column per feature, optional header, rows = time steps) or a .npy
array of shape (T, f).  Values are log returns; pass --prices if the file holds prices.

Pipeline: temporal train/test split -> EVT marginals on train with pooled xi -> PIT ->
CFM training (GPU if available) -> sampling -> inverse PIT -> tail diagnostics,
portfolio VaR/CVaR with bootstrap CIs, Kupiec backtest on the held-out period ->
figures and saved artifacts.

Everything printed is also written to {outdir}/report.log.
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import torch

from csvio import load_returns, feature_names_from_csv
from evt_shrink import shrink_ensemble
from figures import save_all_figures
from run_logging import tee_output
from tailfm import (MarginalEnsemble, VelocityField, train_cfm, sample,
                    estimate_risk, kupiec_test, portfolio_losses, make_windows,
                    print_report)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV or .npy of shape (T, f)")
    ap.add_argument("--prices", action="store_true", help="input is prices, not returns")
    ap.add_argument("--n", type=int, default=24, help="window length")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--q-tail", type=float, default=0.05,
                    help="EVT threshold quantile, per tail")
    ap.add_argument("--nu", type=float, default=5.0,
                    help="degrees of freedom of the t_nu latent space.  The PIT makes "
                         "every marginal exactly t_nu for ANY nu, so this is a design "
                         "parameter rather than an estimate: nu <= 4 gives the CFM "
                         "target x1-x0 an infinite fourth moment and hence "
                         "infinite-variance gradients.  5 is the smallest value with "
                         "E z^4 < inf, and the generated copula is insensitive to it.")
    ap.add_argument("--pos-std", type=float, default=0.1,
                    help="init scale of the positional embedding; controls how much "
                         "within-window volatility clustering the model produces")
    ap.add_argument("--shrink-c", type=float, default=1.0,
                    help="Efron-Morris cap on the xi pooling, in standard errors")
    ap.add_argument("--no-shrink", action="store_true",
                    help="disable pooling of xi across features")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--mix-dim", default="window", choices=["window", "time"],
                    help="coordinates sharing the base mixing variable W")
    ap.add_argument("--gen", type=int, default=20_000, help="# generated windows")
    ap.add_argument("--ode-steps", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--weights", type=str, default=None,
                    help="comma-separated portfolio weights (default: equal)")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--no-recalibrate", action="store_true",
                    help="disable rank-recalibration of generated marginals")
    ap.add_argument("--no-figures", action="store_true",
                    help="skip the diagnostic PNGs")
    ap.add_argument("--no-report", action="store_true",
                    help="skip print_report (Hill / tail dependence / VaR / ACF "
                         "tables).  Neither these nor the PNGs are read by anything "
                         "downstream -- 05/06 and the evaluation scripts need only "
                         "generated_windows.npy -- so both can be dropped in a sweep.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="runs/final")
    ap.add_argument("--log", type=str, default=None,
                    help="text file receiving a copy of everything printed "
                         "(default: {outdir}/report.log)")
    return ap.parse_args()


def run(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    alphas = (0.95, 0.99, 0.995)

    # ------------------------------------------------------------------- data
    r = load_returns(args.data, args.prices)
    T, f = r.shape
    names = feature_names_from_csv(args.data, f)
    split = int((1.0 - args.test_frac) * T)
    train_r, test_r = r[:split], r[split:]
    real = make_windows(train_r, args.n, args.stride)
    print(f"data: T={T}, f={f} | train windows {real.shape} | device={device}")

    w = (np.array([float(v) for v in args.weights.split(",")])
         if args.weights else np.full(f, 1.0 / f))
    assert w.size == f, "--weights length must equal the number of features"

    # -------------------------------------------------- EVT marginals + PIT
    # Fitted on train_r, not on `real`: make_windows overlaps at stride 1, so every
    # interior date would appear n times in the pooled array, inflating the exceedance
    # counts by ~n and breaking the (1+xi)^2/k variance the pooling relies on.
    marg = MarginalEnsemble(q_tail=args.q_tail, nu=args.nu).fit(train_r)
    if not args.no_shrink:
        print("EVT: pooling xi across features (empirical Bayes)")
        shrink_ensemble(marg, train_r, c=args.shrink_c)
    su = marg.summary()
    print(f"EVT: nu={marg.nu_:.2f}, q_tail={args.q_tail}, "
          f"xi_lower median {np.median(su['xi_lo']):+.3f} "
          f"[{su['xi_lo'].min():+.3f}, {su['xi_lo'].max():+.3f}], "
          f"xi_upper median {np.median(su['xi_hi']):+.3f} "
          f"[{su['xi_hi'].min():+.3f}, {su['xi_hi'].max():+.3f}]")
    z = torch.tensor(marg.transform(real), dtype=torch.float32)
    print(f"PIT: max|z| = {np.abs(z.numpy()).max():.1f} (correct fits stay under ~100)")

    # --------------------------------------------------------------- training
    model = VelocityField(f=f, n_max=args.n, d_model=args.d_model, depth=args.depth,
                          pos_std=args.pos_std)
    ema, _ = train_cfm(model, z, nu=marg.nu_, steps=args.steps,
                       batch_size=args.batch, mix_dim=args.mix_dim,
                       device=device, seed=args.seed)
    torch.save(ema.shadow.state_dict(), f"{args.outdir}/model_ema.pt")
    pickle.dump(marg, open(f"{args.outdir}/marginals.pkl", "wb"))

    # --------------------------------------------------------------- sampling
    z_gen = sample(ema.shadow, args.gen, args.n, f, nu=marg.nu_,
                   n_steps=args.ode_steps, mix_dim=args.mix_dim,
                   device=device, seed=args.seed)
    gen = marg.inverse_transform(z_gen.numpy())
    del z_gen
    if not args.no_recalibrate:
        # Rank-recalibration: replace each feature's pooled marginal by exactly F_hat_j
        # via x -> F_hat_j^{-1}(rank/(K+1)).  Leaves the learned copula invariant
        # (Sklar), so the flow keeps only the dependence while the EVT marginals keep
        # the tails.
        flat = gen.reshape(-1, f)
        K = flat.shape[0]
        rank = np.arange(1, K + 1, dtype=float) / (K + 1.0)
        for a in range(0, f, 32):                      # column blocks, cache-friendly
            b = min(a + 32, f)
            blk = np.ascontiguousarray(flat[:, a:b].T)
            idx = np.argsort(blk, axis=1)
            u = np.empty(blk.shape)
            np.put_along_axis(u, idx, np.broadcast_to(rank, blk.shape), axis=1)
            for j in range(a, b):
                flat[:, j] = marg.marginals_[j].ppf(u[j - a])
        gen = flat.reshape(args.gen, args.n, f)
    np.save(f"{args.outdir}/generated_windows.npy", gen)

    # ------------------------------------------------------------ diagnostics
    if not args.no_report:
        print_report(real, gen, feature_names=names)

    # ------------------------------------------------------------ risk report
    report = estimate_risk(gen, alphas=alphas, weights=w, horizon=args.horizon,
                           n_boot=200, seed=args.seed)
    L_test = portfolio_losses(make_windows(test_r, args.horizon, stride=args.horizon),
                              weights=w, horizon=args.horizon)
    print(f"\n=== Portfolio risk (h={args.horizon}) and Kupiec backtest "
          f"(held-out N={L_test.size}) ===")
    for a in alphas:
        rp, k = report[a], kupiec_test(L_test, report[a]["var_gpd"], a)
        print(f"a={a:5.3f}: VaR {rp['var_gpd']:.5f} "
              f"[{rp['var_ci'][0]:.5f},{rp['var_ci'][1]:.5f}]  "
              f"CVaR {rp['cvar_gpd']:.5f} "
              f"[{rp['cvar_ci'][0]:.5f},{rp['cvar_ci'][1]:.5f}]  | "
              f"exceed {k['exceedances']}/{k['expected']:.1f}  p={k['p_value']:.3f}")

    # ----------------------------------------------------------------- figures
    paths = ([] if args.no_figures else
             save_all_figures(real, {"tailfm": gen}, names, args.outdir,
                              weights=w, horizon=args.horizon))
    print(f"\nSaved: {args.outdir}/{{model_ema.pt, marginals.pkl, "
          f"generated_windows.npy}}"
          + ("" if not paths else ", " + ", ".join(os.path.basename(p) for p in paths)))


def main():
    args = parse_args()
    log_path = args.log or f"{args.outdir}/report.log"
    with tee_output(log_path, header="fit_returns.py"):
        run(args)
    print(f"Terminal report saved to {log_path}")


if __name__ == "__main__":
    main()
