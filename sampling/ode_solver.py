from sampling.base_solver import Solver


class EulerSolver(Solver):
    """
    ODE solver, Euler's method
    """

    def step(self, xt, t, labels=None, guidance_scale=None):
        """
        One step of Euler's method
        """
        drift = self.model(xt, t, labels, guidance_scale)
        x_next = xt + drift * self.h
        return x_next


class MidpointSolver(Solver):
    """
    ODE solver, Midpoint method (Runge-Kutta 2nd order)
    """

    def step(self, xt, t, labels=None, guidance_scale=None):
        """
        One step of Midpoint method (RK2)
        """
        k1 = self.model(xt, t, labels, guidance_scale)
        t_mid = t + self.h / 2.0
        xt_mid = xt + (self.h / 2.0) * k1
        k2 = self.model(xt_mid, t_mid, labels, guidance_scale)
        x_next = xt + self.h * k2
        return x_next
