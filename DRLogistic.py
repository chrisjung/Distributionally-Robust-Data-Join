import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, jit
from jax import random
from functools import partial
from typing import NamedTuple
import cvxpy as cp


import matplotlib.pyplot as plt


class ParamsWithAlpha(NamedTuple):
    weights: jnp.ndarray
    bias: jnp.float64
    alpha: jnp.float64


class Projector:
    def __init__(self, d, kappa):
        self.d = d
        self.kappa = kappa

        self.theta = cp.Parameter(d)
        self.new_theta = cp.Variable(d)

        self.alpha = cp.Parameter(1)
        self.new_alpha = cp.Variable(1)

        # 1/2 is here just to compare to the closed form approach, but technically, since we
        # are not using that approach, we can get rid of 1/2.
        self.objective_fcn = 0.5 * (cp.Pnorm(self.theta - self.new_theta, 2) ** 2 + cp.Pnorm(self.new_alpha - self.alpha, 2) ** 2)
        self.constraints = [cp.Pnorm(self.new_theta, 2) <= self.new_alpha, self.new_alpha >= 0]

        self.prob = cp.Problem(cp.Minimize(self.objective_fcn), self.constraints)

    def project(self, theta, alpha, eps=1e-10):
        self.theta.value = np.array(theta)
        self.alpha.value = np.array([alpha])


        self.prob.solve(warm_start=True, feastol=eps)

        return self.new_theta.value, self.new_alpha.value[0]


# TODO: change the current file name from DRODataJoiner2 to DRODataJoiner
class DRLogisticRegression:
    def __init__(self, n_iters=1000, learning_rate=5e-2, random_key=42):
        self.random_key = random.PRNGKey(random_key)

        weight = jnp.array([])
        self.params = ParamsWithAlpha(weight, 0, 0)

        self.n_iters = n_iters
        self.learning_rate = learning_rate

        self.X, self.y = None, None

        self.kappa = None
        self.r = None

    # The first d_X coordinates are the same kinds of features shared between X_A and X_P
    def fit(self, X, y, kappa, r, eps=1e-5, show_convergence=False):
        n, d = X.shape

        self.kappa = kappa
        self.r = r

        my_projector = Projector(d, kappa)

        self.random_key, weights_random_key = random.split(self.random_key, 2)
        init_weights = random.normal(weights_random_key, (d,))
        init_bias = 0.0
        alpha = 1.0

        curr_params_with_alpha = ParamsWithAlpha(init_weights, init_bias, alpha)

        @jit
        def update(params_with_alpha, X, y, kappa):
            # Take derivative with respect to params
            curr_grad = grad(self.cost, argnums=0)(params_with_alpha, X, y, kappa)

            new_params = jax.tree_multimap(
                lambda param, g: param - g * self.learning_rate, params_with_alpha, curr_grad)

            return new_params

        # theta_1, theta_2, alpha_A, alpha_P, eps
        def project(params_with_alpha):
            new_theta, new_alpha = my_projector.project(params_with_alpha.weights, params_with_alpha.alpha)
            return ParamsWithAlpha(new_theta, params_with_alpha.bias, new_alpha)

        if show_convergence:
            cost_hist = []
            for curr_iter in range(self.n_iters):
                curr_params_before_projection = update(curr_params_with_alpha, X, y, kappa)
                curr_params_with_alpha = project(curr_params_before_projection)

                if curr_iter % 20 == 0:
                    curr_cost = self.cost(curr_params_with_alpha, X, y, kappa)
                    cost_hist.append(curr_cost)

            fig = plt.figure()
            plt.plot(cost_hist)
            fig.suptitle('Convergence history')
            plt.xlabel('Iteration')
            plt.ylabel('Cost value')
            plt.show()
        else:
            for curr_iter in range(self.n_iters):
                curr_params_before_projection = update(curr_params_with_alpha, X, y, kappa)
                curr_params_with_alpha = project(curr_params_before_projection)

        return curr_params_with_alpha, self.cost(curr_params_with_alpha, X, y, kappa)

    @partial(jax.jit, static_argnums=(0,))
    def cost(self, params_with_alpha, X, y, kappa):
        n_bar, _ = X.shape

        wX = jnp.dot(X[:, ], params_with_alpha.weights) + params_with_alpha.bias
        ywX = jnp.multiply(y, wX)

        logistic_loss = jnp.logaddexp(jnp.zeros(n_bar), -ywX)
        max_term = jnp.maximum(0, wX - params_with_alpha.alpha * kappa)

        unnormalized_cost = logistic_loss + max_term
        partial_cost = jnp.sum(unnormalized_cost) / n_bar

        total_loss = partial_cost + (params_with_alpha.alpha * self.r)

        return total_loss


    def logistic(self, r):
        return 1 / (1 + jnp.exp(-r))

    def predict(self, X, params):
        return self.logistic(jnp.dot(X, params.weights) + params.bias)
