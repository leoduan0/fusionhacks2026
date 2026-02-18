from desc.io import IOAble
from desc.backend import execute_on_cpu, jnp
from desc.io import load
from desc.plotting import plot_3d
import matplotlib.pyplot as plt
import numpy as np


class FusionHacks2026Submission(IOAble):
    """
    Class for submissions for FusionHacks2026.
    """

    _io_attrs_ = [
        "eq",
        "coilset",
        "P_ext",
    ]

    @execute_on_cpu
    def __init__(
        self,
        eq,
        coilset,
        P_ext,
    ):
        """
        :param eq: desc equilibrium object
        :param coilset: desc CoilSet object
        :param P_ext: float for heating power [MW]
        """
        self.eq = eq
        self.coilset = coilset
        self.P_ext = P_ext

    @property
    def eq(self):
        """float: DESC equilibrium."""
        return self._eq

    @eq.setter
    def eq(self, new):
        self._eq = new

    @property
    def coilset(self):
        """float: DESC CoilSet."""
        return self._coilset

    @coilset.setter
    def coilset(self, new):
        self._coilset = new

    @property
    def P_ext(self):
        """float: Heating power, in MW."""
        return self._P_ext

    @P_ext.setter
    def P_ext(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._P_ext = jnp.float64(float(np.squeeze(new)))


if __name__ == "__main__":
    # Example of saving and loading from the FusionHacks2026Submission class

    eq = load("precise_QA_output.h5")[-1]
    coilset = load("optimized_coilset_scaled.h5")
    P_ext = 10e6

    submission = FusionHacks2026Submission(eq, coilset, P_ext)
    submission.save("submission.h5")

    submission_opened = load("submission.h5")
    fig = plot_3d(submission_opened.eq, "B")
    fig.show()
