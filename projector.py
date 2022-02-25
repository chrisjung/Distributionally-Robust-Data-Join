import numpy as np
import jax.numpy as jnp
import cvxpy as cp

import pdb

class Projector:
    def __init__(self, d_1, d_2, kappa_A, alpha_A_smaller):
        self.theta_1 = cp.Parameter(d_1)
        self.theta_2 = cp.Parameter(d_2)

        self.alpha_A = cp.Parameter(1)
        self.alpha_P = cp.Parameter(1)

        self.new_theta_1 = cp.Variable(d_1)
        self.new_theta_2 = cp.Variable(d_2)

        self.new_alpha_A = cp.Variable(1)
        self.new_alpha_P = cp.Variable(1)

        if alpha_A_smaller:
            # 1/2 is here just to compare to the closed form approach, but technically, since we
            # are not using that approach, we can get rid of 1/2.
            self.objective_fcn = 0.5 * (
                    cp.Pnorm(self.theta_1 - self.new_theta_1, 2) ** 2 + cp.Pnorm(self.theta_2 - self.new_theta_2, 2) ** 2
                    + cp.Pnorm(self.new_alpha_A - self.alpha_A, 2) ** 2 + cp.Pnorm(self.new_alpha_P - self.alpha_P, 2) ** 2)
            self.constraints = [cp.Pnorm(self.new_theta_1, 2) <= self.new_alpha_A + self.new_alpha_P,
                           cp.Pnorm(self.new_theta_2, 2) <= kappa_A * self.new_alpha_A, self.new_alpha_A <= self.new_alpha_P,
                           self.new_alpha_A >= 0, self.new_alpha_P >= 0]
        else:
            self.objective_fcn = 0.5 * (
                    cp.Pnorm(self.theta_1 - self.new_theta_1, 2) ** 2 + cp.Pnorm(self.theta_2 - self.new_theta_2, 2) ** 2
                    + cp.Pnorm(self.new_alpha_A - self.alpha_A, 2) ** 2 + cp.Pnorm(self.new_alpha_P - self.alpha_P, 2) ** 2)
            self.constraints = [cp.Pnorm(self.new_theta_1, 2) <= self.new_alpha_A + self.new_alpha_P,
                                cp.Pnorm(self.new_theta_2, 2) <= kappa_A * self.new_alpha_A,
                                self.new_alpha_A >= self.new_alpha_P,
                                self.new_alpha_A >= 0, self.new_alpha_P >= 0]

        self.prob = cp.Problem(cp.Minimize(self.objective_fcn), self.constraints)

    def project(self, theta_1, theta_2, alpha_A, alpha_P, eps=1e-10):
        self.theta_1.value = np.array(theta_1)
        self.theta_2.value = np.array(theta_2)

        self.alpha_A.value = np.array([alpha_A])
        self.alpha_P.value = np.array([alpha_P])

        self.prob.solve(warm_start=True, feastol=eps)

        return self.new_theta_1.value, self.new_theta_2.value, self.new_alpha_A.value[0], self.new_alpha_P.value[0]

