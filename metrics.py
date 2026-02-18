from desc.equilibrium import equilibrium
from scipy.optimize import brentq, fsolve
import numpy as np
import matplotlib.pyplot as plt
from desc.grid import LinearGrid
from desc.integrals import Bounce2D
from desc.io import load

mu0 = (4 * np.pi) * 1e-7  # permeability of free space
k = 1.38e-23  # boltzmann constant
keV_to_J = 1.6022e-16
E_alpha = 3.5e3 * keV_to_J  # in J
BETA = 0.05


def sigmav(T):
    """
    Compute the reaction rate parameter in units of
    [m^3/s] given the temperature. From Bosch and Hale.

    :param T: Temperature in keV
    Return:
    :sigma_v: Reaction rate parameter [m^3/s]
    """
    # Constants for D-T (T(d,n)4He)
    BG = 34.3827  # sqrt(keV)
    mr_c2 = 1124656  # keV

    C1 = 1.17302e-9
    C2 = 1.51361e-2
    C3 = 7.61886e-2
    C4 = 4.60643e-3
    C5 = 1.35000e-2
    C6 = -1.06750e-4
    C7 = 1.36600e-5

    # Theta(T)
    numerator = T * (C2 + T * (C4 + T * C6))
    denominator = 1.0 + T * (C3 + T * (C5 + T * C7))
    theta = T / (1.0 - (numerator / denominator))

    # xi(T)
    xi = (BG**2 / (4.0 * theta)) ** (1.0 / 3.0)

    # Reactivity
    prefactor = C1 * theta * np.sqrt(xi / (mr_c2 * T**3))
    sigma_v = prefactor * np.exp(-3.0 * xi)
    return 1e-6 * sigma_v


def eps_avg(eq):
    """
    Compute rms epsilon of a given equilibrium
    :param eq: Equilbrium
    Returns:
    :eps: RMS epsilon
    """
    # computing average eps:
    rho = np.linspace(0.01, 1, 5)
    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    X, Y = 16, 32
    theta = Bounce2D.compute_theta(eq, X, Y, rho)
    num_transit = 3
    Y_B = 32
    num_transit = 20
    num_well = 10 * num_transit
    num_quad = 32
    num_pitch = 45
    data = eq.compute(
        "effective ripple",
        grid=grid,
        theta=theta,
        Y_B=Y_B,
        num_transit=num_transit,
        num_well=num_well,
        num_quad=num_quad,
        num_pitch=num_pitch,
        # Can also specify ``pitch_batch_size`` which determines the
        # number of pitch values to compute simultaneously.
        # Reduce this if insufficient memory. If insufficient memory is detected
        # early then the code will exit and return ε = 0 everywhere. If not detected
        # early then typical OOM errors will occur.
    )
    eps = grid.compress(data["effective ripple"])
    return np.sqrt(np.mean(eps**2))


def h_factor(eq, verbose=False):
    """
    Compute the H-factor as per the model in the
    FusionHacks 2026 documentation, given an equilibrium
    :param eq: Equilbrium
    Returns:
    :h: H factor (dimensionless)
    """
    h_max = 2
    h_min = 0.7
    mu = 2
    eps = eps_avg(eq)
    k = 2
    q = eps / 0.1
    h = h_min + ((h_max - h_min) / (1 + np.exp(k * (np.log(q) + mu))))
    if verbose:
        print(f"eps = {eps}")
        print(f"H factor = {h}")
    return h


def tau_E_iss04(h, B, R0, a, n, iota_23, P_ext, verbose=False):
    """
    Compute tau_E_iss04.

    :param h: H factor (dimensionless)
    :param B: Magnetic field on axis [T]
    :param R0: Major radius [m]
    :param a: Minor radius [m]
    :param n: Density [m^-3]
    :param iota_23: iota(rho=2/3) (dimensionless)
    :param P_ext: external power [W]
    :param verbose: Whether to print all variables

    Returns:
    :tau_E: Confinement time [s]
    """
    n0 = n / 1e19
    p0 = P_ext / 1e6
    tau_E = (
        h
        * 0.465
        * (B**0.84)
        * (R0**0.64)
        * (a**2.28)
        * (n0**0.54)
        * (p0 ** (-0.61))
        * (iota_23**0.41)
    )
    if verbose:
        print("B =", B)
        print("R =", R0)
        print("a =", a)
        print("n19 =", n / 1e19)
        print("PMW =", P_ext / 1e6)
        print("iota =", iota_23)
        print("h =", h)
        print("tau_E =", tau_E)
    return tau_E


def pb_res(T, beta, B_on_axis, R0, a, iota_23, h, vol, P_ext):
    """
    Compute the residual of the power balance given dependent
    parameters

    :param T: Temperature [keV]
    :param beta: Plasma beta (dimensionless)
    :param B_on_axis: Magnetic field on axis [T]
    :param R0: Average major radius [m]
    :param a: Average minor radius [m]
    :param iota_23: iota(rho=2/3) (dimensionless)
    :param h: H factor (dimensionless)
    :param vol: Volume [M^3]
    :param P_ext: external power [W]

    Returns:
    :res_mw: Power balance residual in MW
    """

    n = (beta * B_on_axis**2) / (2 * mu0 * T * 1.6022e-16)
    tau_E = tau_E_iss04(h, B_on_axis, R0, a, n, iota_23, P_ext)
    res = (
        P_ext
        + ((n**2) / 4) * sigmav(T) * vol * (E_alpha)
        - 3 * (n * T * vol * 1.6022e-16) / (tau_E)
    )
    res_MW = res / 1e6
    return res_MW


def temp_from_eq(eq: equilibrium, P_ext: float):
    """
    Compute the temperature from the FusionHacks2026 transport
    model power balance.

    :param eq: Equilbrium
    :type eq: desc.equilibrium
    :param P_ext: External power [MW]

    Returns:
    :t_solved: Temperature [keV]
    """
    a = eq.compute("a")["a"]
    R0 = eq.compute("R0")["R0"]
    B_on_axis = eq.compute("<|B|>_axis")["<|B|>_axis"]
    vol = eq.compute("V")["V"]
    rho = [2.0 / 3.0]
    iota_23 = eq.compute("iota", grid=LinearGrid(rho=rho))["iota"]
    h = h_factor(eq, verbose=True)
    beta = BETA
    args = (beta, B_on_axis, R0, a, iota_23, h, vol, P_ext)
    t_solved = fsolve(func=pb_res, args=args, x0=5)
    print(f"T: {t_solved}")
    return t_solved


def neutron_fluence(eq, T):
    """
    Compute the neutron fluence (i.e. total rate of neutron production)!

    :param eq: Equilibrium
    :param T: Temperature [keV]
    Returns:
    :fleunce: Total neutron fluence [neutrons/s]
    """
    beta = BETA
    B_on_axis = eq.compute("<|B|>_axis")["<|B|>_axis"]
    vol = eq.compute("V")["V"]
    n = (beta * B_on_axis**2) / (2 * mu0 * T * 1.6022e-16)
    print(f"density: {n}")
    fluence = 0.25 * (n**2) * sigmav(T) * vol
    return fluence


if __name__ == "__main__":

    # plot <sigma v> for D-T as a function of T

    T = np.logspace(-1, 2, 10)
    sv = sigmav(T)
    plt.loglog(T, sv)
    plt.xlabel("Temperature [keV]")
    plt.ylabel("$\\langle \\sigma v \\rangle$ [$\\text{m}^3/s$]")
    plt.show()

    # compute neutron flux from an example reactor

    eq = load("input.HELIOTRON_output.h5")[-1]
    temp = temp_from_eq(eq, P_ext=10e6)
    print(temp)
    rate = neutron_fluence(eq, temp)
    print(rate)
