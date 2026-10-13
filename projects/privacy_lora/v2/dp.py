"""RDP accountant for the Poisson-subsampled Gaussian mechanism (DP-SGD).

Self-contained port of the standard Mironov et al. (2019) computation used by
TensorFlow-Privacy / Opacus, including the improved RDP->(eps,delta)
conversion of Balle et al. (2020). Used to calibrate the noise multiplier for
DP-LoRA (defense D1) at a target epsilon.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from scipy import special

DEFAULT_ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + [128, 256, 512]


def _log_add(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def _log_sub(a: float, b: float) -> float:
    if b == -math.inf:
        return a
    if a == b:
        return -math.inf
    return a + math.log1p(-math.exp(b - a))


def _log_erfc(x: float) -> float:
    return math.log(2) + special.log_ndtr(-x * 2**0.5)


def _log_a_int(q: float, sigma: float, alpha: int) -> float:
    log_a = -math.inf
    for i in range(alpha + 1):
        log_coef = math.lgamma(alpha + 1) - math.lgamma(i + 1) - math.lgamma(alpha - i + 1)
        s = log_coef + i * math.log(q) + (alpha - i) * math.log(1 - q) + (i * i - i) / (2 * sigma**2)
        log_a = _log_add(log_a, s)
    return log_a


def _log_a_frac(q: float, sigma: float, alpha: float) -> float:
    log_a0, log_a1 = -math.inf, -math.inf
    i = 0
    z0 = sigma**2 * math.log(1 / q - 1) + 0.5
    while True:
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i
        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)
        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))
        log_s0 = log_t0 + (i * i - i) / (2 * sigma**2) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * sigma**2) + log_e1
        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)
        i += 1
        if max(log_s0, log_s1) < -30:
            break
    return _log_add(log_a0, log_a1)


def _rdp_one(q: float, sigma: float, alpha: float) -> float:
    if q == 0:
        return 0.0
    if q == 1.0:
        return alpha / (2 * sigma**2)
    if math.isinf(alpha):
        return math.inf
    if float(alpha).is_integer():
        return _log_a_int(q, sigma, int(alpha)) / (alpha - 1)
    return _log_a_frac(q, sigma, alpha) / (alpha - 1)


def compute_rdp(q: float, sigma: float, steps: int, orders: Sequence[float] = DEFAULT_ORDERS):
    return [steps * _rdp_one(q, sigma, a) for a in orders]


def rdp_to_eps(orders: Sequence[float], rdp: Sequence[float], delta: float) -> Tuple[float, float]:
    best_eps, best_order = math.inf, None
    for a, r in zip(orders, rdp):
        if a <= 1:
            continue
        eps = r - (math.log(delta) + math.log(a)) / (a - 1) + math.log((a - 1) / a)
        if eps < best_eps:
            best_eps, best_order = eps, a
    return max(best_eps, 0.0), best_order


def epsilon_for(sigma: float, q: float, steps: int, delta: float) -> float:
    rdp = compute_rdp(q, sigma, steps)
    return rdp_to_eps(DEFAULT_ORDERS, rdp, delta)[0]


def noise_multiplier_for(target_eps: float, q: float, steps: int, delta: float, tol: float = 0.01) -> float:
    """Smallest sigma giving epsilon <= target_eps (bisection)."""
    lo, hi = 0.2, 200.0
    if epsilon_for(hi, q, steps, delta) > target_eps:
        raise ValueError("target epsilon unreachable with sigma<=200")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if epsilon_for(mid, q, steps, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


if __name__ == "__main__":
    # sanity: matches Opacus values, e.g. q=0.01, sigma=1.0, 10k steps, delta=1e-5 -> eps ~ 1.6
    print("eps", epsilon_for(1.0, 0.01, 10_000, 1e-5))
    n, batch, epochs = 1000, 256, 20
    q = batch / n
    steps = int(epochs / q)
    delta = 1.0 / n**1.1
    for eps in (1, 3, 8):
        print(eps, noise_multiplier_for(eps, q, steps, delta))
