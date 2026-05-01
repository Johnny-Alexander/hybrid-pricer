"""
Closed-form bivariate Black-Scholes pricer for an EQ/FX hybrid:

    Payoff at T = max(S_T - K, 0) * 1{X_T > B}

where S = equity (e.g. SPX), X = FX rate (e.g. USDJPY), B = FX barrier,
K = equity strike. Settlement currency is the domestic currency (USD).

Model
-----
Under the domestic risk-neutral measure Q^d:
    dS/S = (r_d - q) dt + sigma_S dW_S
    dX/X = (r_d - r_f) dt + sigma_X dW_X
    corr(dW_S, dW_X) = rho

Closed-form price
-----------------
    V0 = S0 e^{-qT}  M2(d1S, d2X + rho*sigma_S*sqrt(T); rho)
       - K  e^{-r_d T} M2(d2S, d2X; rho)

with
    d1S = [ln(S0/K) + (r_d - q + 0.5*sigma_S^2)*T] / (sigma_S*sqrt(T))
    d2S = d1S - sigma_S*sqrt(T)
    d1X = [ln(X0/B) + (r_d - r_f + 0.5*sigma_X^2)*T] / (sigma_X*sqrt(T))
    d2X = d1X - sigma_X*sqrt(T)

The first term arises from a change of numeraire to S; the FX argument
shifts by rho*sigma_S*sqrt(T) under that measure.

References
----------
- Heynen & Kat (1994), "Crossing Barriers"
- Haug, "Complete Guide to Option Pricing Formulas", Ch. on multi-asset options
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, multivariate_normal


def bivariate_normal_cdf(a: float, b: float, rho: float) -> float:
    """P(Z1 <= a, Z2 <= b) where (Z1, Z2) is standard bivariate normal with
    correlation rho."""
    return multivariate_normal(mean=[0.0, 0.0],
                               cov=[[1.0, rho], [rho, 1.0]]).cdf([a, b])


@dataclass
class HybridInputs:
    """Inputs for the hybrid pricer.

    Attributes
    ----------
    S0     : Equity spot
    X0     : FX spot (units: domestic per foreign, e.g. USDJPY)
    K      : Equity strike
    B      : FX barrier (option pays only if X_T > B)
    T      : Time to expiry in years
    r_d    : Domestic risk-free rate (continuous)
    r_f    : Foreign risk-free rate (continuous)
    q      : Equity dividend yield (continuous)
    sig_S  : Equity volatility
    sig_X  : FX volatility
    rho    : Correlation between dW_S and dW_X under domestic measure
    """
    S0: float
    X0: float
    K: float
    B: float
    T: float
    r_d: float
    r_f: float
    q: float
    sig_S: float
    sig_X: float
    rho: float


def price_hybrid_call(p: HybridInputs) -> dict:
    """Closed-form price of the SPX-call x FX-digital hybrid.

    Returns
    -------
    dict with keys:
        price                  : option fair value (per equity unit)
        term1, term2           : the two components of the closed form
        P_joint_exercise       : Q-probability of joint event {S_T>K, X_T>B}
        P_FX_above_barrier     : marginal Q-probability of {X_T>B}
        vanilla_SPX_call       : price of the unconditional SPX call (for comparison)
    """
    sqrtT = np.sqrt(p.T)

    d1S = (np.log(p.S0 / p.K) + (p.r_d - p.q + 0.5 * p.sig_S ** 2) * p.T) / (p.sig_S * sqrtT)
    d2S = d1S - p.sig_S * sqrtT

    d1X = (np.log(p.X0 / p.B) + (p.r_d - p.r_f + 0.5 * p.sig_X ** 2) * p.T) / (p.sig_X * sqrtT)
    d2X = d1X - p.sig_X * sqrtT

    # Joint event {S_T > K, X_T > B} under each measure.
    # Under Q^d:  P = M2(d2S, d2X; rho)  (using the upper-tail symmetry)
    # Under Q^S:  the FX log-mean shifts by rho*sig_S*sig_X*T, so the FX
    #             argument becomes d2X + rho*sig_S*sqrt(T).
    term1 = p.S0 * np.exp(-p.q * p.T) * bivariate_normal_cdf(
        d1S, d2X + p.rho * p.sig_S * sqrtT, p.rho
    )
    term2 = p.K * np.exp(-p.r_d * p.T) * bivariate_normal_cdf(d2S, d2X, p.rho)

    price = term1 - term2

    P_joint = bivariate_normal_cdf(d2S, d2X, p.rho)
    P_fx = norm.cdf(d2X)
    vanilla = (p.S0 * np.exp(-p.q * p.T) * norm.cdf(d1S)
               - p.K * np.exp(-p.r_d * p.T) * norm.cdf(d2S))

    return {
        "price": price,
        "term1": term1,
        "term2": term2,
        "P_joint_exercise": P_joint,
        "P_FX_above_barrier": P_fx,
        "vanilla_SPX_call": vanilla,
    }


def price_hybrid_call_mc(p: HybridInputs, n_paths: int = 2_000_000,
                         seed: int = 42) -> dict:
    """Monte Carlo cross-check using terminal joint lognormal draws."""
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n_paths)
    z_ind = rng.standard_normal(n_paths)
    z2 = p.rho * z1 + np.sqrt(1.0 - p.rho ** 2) * z_ind

    sqrtT = np.sqrt(p.T)
    S_T = p.S0 * np.exp((p.r_d - p.q - 0.5 * p.sig_S ** 2) * p.T + p.sig_S * sqrtT * z1)
    X_T = p.X0 * np.exp((p.r_d - p.r_f - 0.5 * p.sig_X ** 2) * p.T + p.sig_X * sqrtT * z2)

    payoff = np.maximum(S_T - p.K, 0.0) * (X_T > p.B)
    pv = np.exp(-p.r_d * p.T) * payoff
    price = pv.mean()
    se = pv.std(ddof=1) / np.sqrt(n_paths)
    return {
        "price": price,
        "std_error": se,
        "ci95": (price - 1.96 * se, price + 1.96 * se),
    }


def _bump(p: HybridInputs, field: str, h: float) -> float:
    """Return price with one input bumped by h."""
    d = p.__dict__.copy()
    d[field] = d[field] + h
    return price_hybrid_call(HybridInputs(**d))["price"]


def greeks(p: HybridInputs) -> dict:
    """Bump-and-revalue Greeks. Bumps are sized for stability.

    Returns
    -------
    dict with delta_SPX, gamma_SPX, delta_FX, vega_SPX, vega_FX, cega_corr,
    rho_USD, rho_JPY (all in natural units per inputs).
    """
    base = price_hybrid_call(p)["price"]

    dS = 0.01 * p.S0
    dX = 0.01 * p.X0
    dv = 0.01    # 1 vol point
    dr = 1e-4    # 1bp
    drho = 0.01

    return {
        "delta_SPX": (_bump(p, "S0", dS) - _bump(p, "S0", -dS)) / (2 * dS),
        "gamma_SPX": (_bump(p, "S0", dS) - 2 * base + _bump(p, "S0", -dS)) / (dS ** 2),
        "delta_FX": (_bump(p, "X0", dX) - _bump(p, "X0", -dX)) / (2 * dX),
        "vega_SPX": (_bump(p, "sig_S", dv) - _bump(p, "sig_S", -dv)) / (2 * dv) / 100,
        "vega_FX":  (_bump(p, "sig_X", dv) - _bump(p, "sig_X", -dv)) / (2 * dv) / 100,
        "cega_corr": (_bump(p, "rho", drho) - _bump(p, "rho", -drho)) / (2 * drho) / 100,
        "rho_USD": (_bump(p, "r_d", dr) - _bump(p, "r_d", -dr)) / (2 * dr) / 10000,
        "rho_JPY": (_bump(p, "r_f", dr) - _bump(p, "r_f", -dr)) / (2 * dr) / 10000,
    }
