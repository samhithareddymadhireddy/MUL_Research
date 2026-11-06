"""Privacy accounting for noise-injected approximate unlearning.

Regenerates Table I and Table VII of

    "Verifying the Right to Be Forgotten in Deployed Chest X-Ray Models:
     Confounded Metrics, Silent Rare-Disease Degradation, and a Calibrated
     Audit Protocol"

from (eta, C, B, E, sigma, delta, k) alone. No GPU, no data, no model.

    python privacy_accounting.py

Four accountings, side by side:

  classical Gaussian   sigma >= Delta2 * sqrt(2 ln(1.25/delta)) / eps
                       Valid ONLY for eps <= 1. Outside that range it does not
                       merely lose tightness, it ceases to hold -- and at our
                       aggregate sensitivity it returns values SMALLER than the
                       exact condition (48.5 against 91.8 at sigma=1e-4), i.e.
                       it understates the privacy loss. Included for comparison
                       with prior work only; do not report it.

  analytic Gaussian    Balle & Wang, ICML 2018. Exact, necessary and
                       sufficient, no restriction on eps. Use this.

  advanced composition k applications of an (eps0, delta0) mechanism. Stated
                       for ARBITRARY mechanisms, so the k*eps0*(e^eps0 - 1)
                       term dominates as soon as eps0 approaches 1. Applying it
                       to Gaussian noise over-prices the budget by roughly an
                       order of magnitude.

  zCDP                 Bun & Steinke, TCC 2016. The Gaussian mechanism is
                       rho-zCDP with rho = Delta2^2 / 2 sigma^2, and zCDP
                       composes ADDITIVELY, which is what the Renyi divergence
                       of Gaussian noise actually supports. Correct instrument.

The sensitivity premise matters more than the accounting. Under per-sample
clipping at threshold C, one record moves the accumulated update by at most
2*eta*C*E/B. Reference implementations instead call clip_grad_norm_ on the
AGGREGATE batch gradient, which bounds no individual contribution and inflates
the sensitivity by exactly the batch size B. Both regimes are reported below;
the aggregate one is what the published code does, and no noise scale
preserving diagnostic utility gives it a non-vacuous budget.

Every epsilon obtained this way is CONDITIONAL on fixed theta. It bounds the
sensitivity of the unlearning update, not the release, since theta itself was
trained on the forget set. It is not an end-to-end unlearning guarantee.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr
from scipy.stats import norm

# --------------------------------------------------------------------------
# Paper configuration
# --------------------------------------------------------------------------
ETA = 1e-4      # unlearning learning rate
CLIP = 1.0      # gradient clipping threshold C
BATCH = 32      # batch size B
EPOCHS = 5      # unlearning epochs E
DELTA = 1e-5    # delta in (eps, delta)
DELTA_P = 1e-5  # delta' slack term in advanced composition
REQUESTS = 15   # deletion requests over the deployment lifetime
SIGMAS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


def sensitivity(eta=ETA, C=CLIP, B=BATCH, E=EPOCHS, per_sample=True):
    """L2 sensitivity of the accumulated update to one forget record.

    Proposition 1. Per-sample clipping gives 2*eta*C*E/B; aggregate clipping
    bounds only the summed gradient, so one record may account for the whole
    clipped norm and the bound inflates by a factor of B.
    """
    delta2 = 2.0 * eta * C * E / B
    return delta2 if per_sample else B * delta2


def eps_classical(delta2, sigma, delta=DELTA):
    """Classical Gaussian-mechanism bound. Only meaningful for eps <= 1."""
    return delta2 * np.sqrt(2.0 * np.log(1.25 / delta)) / sigma


def eps_analytic(delta2, sigma, delta=DELTA):
    """Exact epsilon by the analytic Gaussian mechanism (Balle & Wang 2018).

    Solves delta(eps) = Phi(D/2s - eps*s/D) - e^eps * Phi(-D/2s - eps*s/D)
    for eps. The second term is evaluated in log space; without that,
    e^eps * Phi(...) overflows past eps ~ 700, which is reachable at the
    aggregate sensitivity.
    """
    def residual(eps):
        a = delta2 / (2.0 * sigma) - eps * sigma / delta2
        b = -delta2 / (2.0 * sigma) - eps * sigma / delta2
        return norm.cdf(a) - np.exp(eps + log_ndtr(b)) - delta

    hi = 1.0
    while residual(hi) > 0.0:
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("no bracket found; check delta2 / sigma")
    return brentq(residual, 1e-13, hi, xtol=1e-14, rtol=1e-14)


def compose_advanced(eps0, k=REQUESTS, delta_p=DELTA_P):
    """Advanced composition. Correct but loose for Gaussian mechanisms.

    The exp term overflows for eps0 beyond about 700, which the aggregate
    sensitivity reaches at small sigma. Returning inf there is the honest
    answer: the budget is vacuous, and the exact magnitude is not meaningful.
    """
    if eps0 > 700.0:
        return np.inf
    return (np.sqrt(2.0 * k * np.log(1.0 / delta_p)) * eps0
            + k * eps0 * (np.exp(eps0) - 1.0))


def compose_zcdp(delta2, sigma, k=REQUESTS, delta=DELTA):
    """zCDP composition (Bun & Steinke 2016). Returns (eps_tot, k*rho)."""
    rho = k * delta2 ** 2 / (2.0 * sigma ** 2)
    return rho + 2.0 * np.sqrt(rho * np.log(1.0 / delta)), rho


def sigma_for_budget(delta2, target=1.0, k=REQUESTS, delta=DELTA,
                     method="zcdp"):
    """Smallest noise scale holding eps_tot <= target over k requests."""
    if method == "zcdp":
        f = lambda s: compose_zcdp(delta2, s, k, delta)[0] - target
    elif method == "analytic":
        f = lambda s: compose_advanced(eps_analytic(delta2, s, delta), k) - target
    elif method == "classical":
        f = lambda s: compose_advanced(eps_classical(delta2, s, delta), k) - target
    else:
        raise ValueError("method must be zcdp, analytic or classical")

    # Bracket upward from a scale where the budget is certainly loose. Starting
    # at 1e-8 puts eps0 in the thousands, where the analytic solver has no
    # bracket and advanced composition is already inf.
    lo, hi = 1e-5, 1e-5
    while f(hi) > 0.0:
        hi *= 2.0
        if hi > 1e3:
            return float("nan")
    while f(lo) < 0.0:
        lo /= 2.0
        if lo < 1e-12:
            return float("nan")
    return brentq(f, lo, hi, xtol=1e-14, rtol=1e-14)


def main():
    print("Privacy accounting for noise-injected approximate unlearning")
    print(f"\neta={ETA:g}  C={CLIP:g}  B={BATCH}  E={EPOCHS}  "
          f"delta={DELTA:g}  k={REQUESTS}\n")

    for label, per_sample in (("PER-SAMPLE clipping (Prop. 1)", True),
                              ("AGGREGATE clipping (as implemented)", False)):
        d2 = sensitivity(per_sample=per_sample)
        print("=" * 78)
        print(f"{label}    Delta2 = {d2:.4g}")
        print("=" * 78)
        print(f"{'sigma':>9} {'eps0 cls':>10} {'eps0 ana':>10} "
              f"{'tighter':>8} {'adv comp':>12} {'zCDP':>12}")
        for s in SIGMAS:
            ec = eps_classical(d2, s)
            ea = eps_analytic(d2, s)
            tighter = f"{100 * (1 - ea / ec):.0f}%" if ea < ec else "--"
            ez, _ = compose_zcdp(d2, s)
            print(f"{s:>9.0e} {ec:>10.4g} {ea:>10.4g} {tighter:>8} "
                  f"{compose_advanced(ea):>12.4g} {ez:>12.4g}")

        print(f"\n  sigma* for eps_tot <= 1 over {REQUESTS} requests:")
        for m in ("zcdp", "analytic", "classical"):
            v = sigma_for_budget(d2, method=m)
            print(f"    {m:<10} {'not reachable' if np.isnan(v) else f'{v:.4g}'}")
        print()

    print("=" * 78)
    print("As reported in the paper")
    print("=" * 78)
    d2, d2a = sensitivity(), sensitivity(per_sample=False)
    print(f"  Table I,   per-sample : eps0 = {eps_analytic(d2, 1e-4):.3f}, "
          f"eps_tot(zCDP) = {compose_zcdp(d2, 1e-4)[0]:.2f}")
    print(f"  Table I,   aggregate  : eps0 = {eps_analytic(d2a, 1e-4):.1f}, "
          f"eps_tot(zCDP) = {compose_zcdp(d2a, 1e-4)[0]:.0f}")
    print(f"  Table VII, sigma=1e-3 : eps_tot(zCDP) = "
          f"{compose_zcdp(d2, 1e-3)[0]:.3f} at macro AUC 0.8021,")
    print("    above the 0.7975 measured at the sigma=1e-4 in common use --")
    print("    the lifetime budget is free once the clipping is fixed.")


if __name__ == "__main__":
    main()