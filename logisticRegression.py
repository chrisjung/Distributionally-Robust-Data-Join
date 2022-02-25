import jax.numpy as jnp
import jax
from jax import grad, jit
from jax import random


from jax.experimental import optimizers

from functools import partial


import matplotlib.pyplot as plt

from typing import NamedTuple

jax.config.update('jax_enable_x64', True)



class Params(NamedTuple):
    weights: jnp.ndarray
    bias: jnp.ndarray


class LogisticRegression:
    def __init__(self, n_iters=1000, learning_rate=5e-2, random_key=42):
        self.random_key = random.PRNGKey(random_key)

        weight = jnp.array([])
        bias = jnp.array([])
        self.params = Params(weight, bias)

        self.n_iters = n_iters
        self.learning_rate = learning_rate

    def fit(self, X, y, show_convergence=False):
        n, d = X.shape

        # set the initial params
        self.random_key, weights_random_key, bias_random_key = random.split(self.random_key, 3)
        new_weights = random.normal(weights_random_key, (d,))
        new_bias = random.normal(bias_random_key, (1,))
        self.params = Params(new_weights, new_bias)

        X = jnp.array(X)
        y = jnp.array(y)

        @jit
        def step(params, X, y):
            # Take derivative with respect to params
            curr_grad = grad(self.cost, argnums=0)(params, X, y)
            new_params = jax.tree_multimap(
                lambda param, g: param - g * self.learning_rate, params, curr_grad)

            return new_params

        if show_convergence:
            cost_hist = []
            for i in range(self.n_iters):
                self.params = step(self.params, X, y)
                curr_cost = self.cost(self.params, X, y)
                cost_hist.append(curr_cost)

            fig = plt.figure()
            plt.plot(cost_hist)
            fig.suptitle('Convergence history')
            plt.xlabel('Iteration')
            plt.ylabel('Cost value')
            plt.show()
        else:
            for i in range(self.n_iters):
                self.params = step(self.params, X, y)

    def fit_regularized(self, X, y, regularized_penalty=1, show_convergence=False):
        n, d = X.shape

        # set the initial params
        self.random_key, weights_random_key, bias_random_key = random.split(self.random_key, 3)
        new_weights = random.normal(weights_random_key, (d,))
        new_bias = random.normal(bias_random_key, (1,))
        self.params = Params(new_weights, new_bias)

        X = jnp.array(X)
        y = jnp.array(y)

        @jit
        def step(params, X, y):
            # Take derivative with respect to params
            curr_grad = grad(self.cost_regularized, argnums=0)(params, X, y, regularized_penalty)
            new_params = jax.tree_multimap(
                lambda param, g: param - g * self.learning_rate, params, curr_grad)

            return new_params

        if show_convergence:
            cost_hist = []
            for i in range(self.n_iters):
                self.params = step(self.params, X, y)
                curr_cost = self.cost(self.params, X, y)
                cost_hist.append(curr_cost)

            fig = plt.figure()
            plt.plot(cost_hist)
            fig.suptitle('Convergence history')
            plt.xlabel('Iteration')
            plt.ylabel('Cost value')
            plt.show()
        else:
            for i in range(self.n_iters):
                self.params = step(self.params, X, y)

    def fit_adam(self, X, y, show_convergence=False):
        n, d = X.shape

        # set the initial params
        self.random_key, weights_random_key, bias_random_key = random.split(self.random_key, 3)
        new_weights = random.normal(weights_random_key, (d,))
        new_bias = random.normal(bias_random_key, (1,))
        self.params = Params(new_weights, new_bias)

        X = jnp.array(X)
        y = jnp.array(y)

        opt_init, opt_update, opt_get_params = optimizers.adagrad(self.learning_rate)

        curr_opt_state = opt_init(self.params)

        @jit
        def step(t, opt_state):
            curr_params = opt_get_params(opt_state)
            curr_grad = grad(self.cost, argnums=0)(curr_params, X, y)
            return opt_update(t, curr_grad, opt_state)

        if show_convergence:
            cost_hist = []
            for t in range(self.n_iters):
                curr_opt_state = step(t, curr_opt_state)
                self.params = opt_get_params(curr_opt_state)
                curr_cost = self.cost(self.params, X, y)
                cost_hist.append(curr_cost)

            fig = plt.figure()
            plt.plot(cost_hist)
            fig.suptitle('Convergence history')
            plt.xlabel('Iteration')
            plt.ylabel('Cost value')
            plt.show()
        else:
            for t in range(self.n_iters):
                curr_opt_state = step(t, curr_opt_state)
                self.params = opt_get_params(curr_opt_state)

    def logistic(self, r):
        return 1 / (1 + jnp.exp(-r))

    def predict(self, X, params):
        return self.logistic(jnp.dot(X, params.weights) + params.bias)

    @partial(jax.jit, static_argnums=(0,))
    def cost(self, params, X, y):
        n = y.size
        wX_plus_b = jnp.dot(X, params.weights) + params.bias
        yWX_plus_b = jnp.multiply(y, wX_plus_b)

        # using logaddexp in order to avoid overflow/underflow issues
        unnormalized_cost = jnp.logaddexp(jnp.zeros(n), -yWX_plus_b)

        return jnp.sum(unnormalized_cost) / n

    @partial(jax.jit, static_argnums=(0,))
    def cost_regularized(self, params, X, y, regularized_penalty):
        n = y.size
        wX_plus_b = jnp.dot(X, params.weights) + params.bias
        yWX_plus_b = jnp.multiply(y, wX_plus_b)

        # using logaddexp in order to avoid overflow/underflow issues
        unnormalized_cost = jnp.logaddexp(jnp.zeros(n), -yWX_plus_b)

        return jnp.sum(unnormalized_cost) / n + regularized_penalty * jnp.linalg.norm(params.weights, 2)

