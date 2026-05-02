# kts/cpd_auto.py
"""
Automatic change-point detection with penalty-based model selection.
"""
import numpy as np
from kts.cpd_nonlin import cpd_nonlin


def cpd_auto(K, ncp, vmax, desc_rate=1, **kwargs):
    """Detect change points, automatically selecting their number.

    :param K: Kernel matrix between frame pairs.
    :param ncp: Maximum number of change points.
    :param vmax: Special parameter for penalty computation.
    :param desc_rate: Descriptor sampling rate.
    :param kwargs: Extra parameters for cpd_nonlin.
    :return: (cps, costs) — selected change-points and their costs.
    """
    m = ncp
    _, scores = cpd_nonlin(K, m, backtrack=False, **kwargs)

    N = K.shape[0]
    N2 = N * desc_rate

    penalties = np.zeros(m + 1)
    ncp_range = np.arange(1, m + 1)
    penalties[1:] = (vmax * ncp_range / (2.0 * N2)) * (
        np.log(float(N2) / ncp_range) + 1
    )

    costs = scores / float(N) + penalties
    m_best = np.argmin(costs)
    cps, scores2 = cpd_nonlin(K, m_best, **kwargs)

    return cps, scores2
