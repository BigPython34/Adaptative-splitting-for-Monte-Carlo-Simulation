#!/usr/bin/env python3
"""Core mathematical functions for SMC analysis."""

import numpy as np
from typing import Tuple, Optional, List

try:
    from .config import CORE_CONSTANTS
except ImportError:
    from script.smc.config import CORE_CONSTANTS


def phi(x: np.ndarray) -> np.ndarray:
    """Score function phi(x) = -x for rare event simulation."""
    return CORE_CONSTANTS["phi_multiplier"] * x


def mcmc_kernel(
    x: float, L_current: float, n_steps: int, sigma: float, return_trace: bool = False
) -> Tuple[float, float, Optional[List[float]]]:
    """
    Metropolis-Hastings kernel to sample from N(0,1) truncated at x <= -L_current.

    Args:
        x: Initial particle value
        L_current: Current threshold (constraint x <= -L_current)
        n_steps: Number of MCMC steps
        sigma: Standard deviation for proposals
        return_trace: If True, returns full trace

    Returns:
        Tuple of (final_x, acceptance_rate, trace)
    """
    x_current = x
    accepts = 0
    trace = [] if return_trace else None

    for _ in range(n_steps):
        proposal = x_current + np.random.normal(
            CORE_CONSTANTS["mcmc_proposal_mean"], sigma
        )

        if proposal <= -L_current:
            log_ratio = CORE_CONSTANTS["log_ratio_coefficient"] * (
                proposal**2 - x_current**2
            )
            alpha = np.exp(log_ratio)

            if np.random.rand() < alpha:
                x_current = proposal
                accepts += 1

        if return_trace:
            trace.append(x_current)

    acceptance_rate = (
        accepts / n_steps if n_steps > 0 else CORE_CONSTANTS["acceptance_default"]
    )

    if return_trace:
        return x_current, acceptance_rate, trace
    else:
        return x_current, acceptance_rate, None
