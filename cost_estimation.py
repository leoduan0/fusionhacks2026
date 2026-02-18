from desc.io import load
import desc
import numpy as np


def total_reactor_cost(eq, coilset, P_ext):
    """
    Compute initial reactor cost for FusionHacks2026

    :param eq: equilibrium
    :param coilset: coilset
    :param P_ext: heating power [MW]
    """
    vol = eq.compute("V")["V"]
    total_current_length = 0
    mean_curvature_list = []
    mean_torsion_list = []
    max_curvature_list = []
    max_torsion_list = []
    for coil in coilset:
        total_current_length += coil.current * coil.compute("length")["length"]
        kappa = coil.compute("curvature")["curvature"]
        tau = coil.compute("torsion")["torsion"]
        mean_curvature_list.append(np.mean(kappa**2))
        mean_torsion_list.append(np.mean(tau**2))
        max_curvature_list.append(np.max(kappa**2))
        max_torsion_list.append(np.max(tau**2))

    mean_k2 = np.mean(mean_curvature_list)
    mean_t2 = np.mean(mean_torsion_list)
    max_k2 = np.max(max_curvature_list)
    max_t2 = np.max(max_torsion_list)

    vol0 = 50
    l0 = 1e7
    kt0 = 5
    p0 = 10e6

    c_v = 0.4
    c_coils = 0.5
    c_l = 0.6
    c_kt = 0.4
    f_k = 0.99
    f_t = 1e-3
    c_p = 0.1

    baseline = 5e8

    cost_kt = np.sqrt(f_k * mean_k2 + f_t * mean_t2) + np.sqrt(
        f_k * max_k2 + f_t * max_t2
    )  # cost due to curvature and torsion
    cost = baseline * (
        c_v * (vol / vol0) ** 1.1
        + c_coils
        * (
            c_l * np.abs(total_current_length / l0) ** 1.2
            + c_kt * (cost_kt / kt0) ** 1.4
        )
        + c_p * P_ext / p0
    )
    return cost
