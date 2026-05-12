"""Centralised default hyper-parameters for Phase 1 (Zhang et al. 2016).

All magic-number defaults for the CnnColor model are defined here.
Import these constants wherever a default value is needed — never hard-code
them in argparse, function signatures, or fallback expressions.
"""

# ── annealed-mean temperature ────────────────────────────────────────────
# Paper value: 0.38.  Used in decode_zhang2016_annealed_mean().
T = 0.38

# ── prior-mix rebalance gamma ────────────────────────────────────────────
# Paper value: 0.5.  Used in build_zhang2016_rebalance_weights().
REBALANCE_GAMMA = 0.5
