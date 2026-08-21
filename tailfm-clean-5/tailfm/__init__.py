"""tailfm: tail-aware flow matching for multivariate return generation, targeted at
VaR / CVaR estimation.

Pipeline: EVT marginal PIT (evt) -> CFM with Student-t base (base, model, cfm)
-> inverse PIT -> risk estimation (risk) and tail diagnostics (evaluate).
"""

from .evt import MarginalEnsemble, SemiParametricMarginal, hill_estimator
from .base import sample_base
from .model import VelocityField
from .cfm import train_cfm, sample, EMA
from .risk import estimate_risk, kupiec_test, portfolio_losses, var_cvar_empirical, var_cvar_gpd
from .evaluate import print_report, hill_table, tail_dependence_report, marginal_risk_table, acf
from .data import make_windows
