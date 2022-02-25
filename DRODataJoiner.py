import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, jit
from jax import random
from functools import partial
from typing import NamedTuple
import itertools
from sklearn.neighbors import KDTree

import projector
import matplotlib.pyplot as plt


class ParamsWithAlphas(NamedTuple):
    weights: jnp.ndarray
    bias: jnp.float64
    alpha_A: jnp.float64
    alpha_P: jnp.float64


class DRODataJoiner:
    def __init__(self, n_iters=1000, learning_rate=5e-2, random_key=42):
        self.random_key = random.PRNGKey(random_key)

        weight = jnp.array([])
        self.params = ParamsWithAlphas(weight, 0, 0, 0)

        self.n_iters = n_iters
        self.learning_rate = learning_rate

        self.X_A, self.X_P = None, None

        self.X_bar_A = None
        self.X_bar_P = None
        self.y_bar = None

        self.pairwise_distance_avg = None
        self.kappa_A = None
        self.kappa_P = None
        self.r_A = None
        self.r_P = None
        self.d_X = None

        self.matchings = None

    def set_k_closest_matchings(self, k=5):
        n_A, _ = self.X_A[:, :self.d_X].shape
        n_P, _ = self.X_P.shape

        # Handle X_A's
        k_closest_neighbors_A = KDTree(self.X_A[:, :self.d_X])
        neighbors_from_A = k_closest_neighbors_A.query(self.X_P[:, :self.d_X], k, return_distance=False)

        # Handle X_P's
        k_closest_neighbors_P = KDTree(self.X_P)
        neighbors_from_P = k_closest_neighbors_P.query(self.X_A[:, :self.d_X], k, return_distance=False)

        matchings_list = []
        for (j, neighbors) in enumerate(neighbors_from_A):
            [matchings_list.append((neighbor, j)) for neighbor in neighbors]

        for (i, neighbors) in enumerate(neighbors_from_P):
            [matchings_list.append((i, neighbor)) for neighbor in neighbors]

        matchings = np.array(list(set(matchings_list)))

        return matchings

    def initialize(self, X_A, X_P, y_P, kappa_A, kappa_P, r_A, r_P, num_nearest_neighbors=3):
        self.X_A, self.X_P = X_A, X_P
        # Anything that's not a variable (things that don't get affected as we change alpha_A, alpha_P, thetas)
        # for the optimization gets saved as attributes
        # which X_bar_A, X_bar_P gets used changes as we change alpha_A, alpha_P, but the actual content doesn't change
        n_A, _ = X_A.shape
        n_P, self.d_X = X_P.shape

        if num_nearest_neighbors:
            self.matchings = self.set_k_closest_matchings(num_nearest_neighbors)
        else:
            self.matchings = itertools.product(range(n_A), range(n_P))

        self.X_bar_A, self.y_bar = self.get_X_bar_y_bar_A(X_A, y_P)
        self.X_bar_P, _ = self.get_X_bar_y_bar_P(X_A, X_P, y_P, self.d_X)

        self.pairwise_distance_avg = self.get_pairwise_distance_avg(X_A, X_P, self.d_X)
        self.kappa_A, self.kappa_P, self.r_A, self.r_P = kappa_A, kappa_P, r_A, r_P

        self.y_bar = self.y_bar

    # The first d_X coordinates are the same kinds of features shared between X_A and X_P
    def fit(self, X_A, X_P, y_P, kappa_A, kappa_P, r_A, r_P, eps=1e-5, num_nearest_neighbors=3, show_convergence=False):
        # Must have been initialized
        self.initialize(X_A, X_P, y_P, kappa_A, kappa_P, r_A, r_P, num_nearest_neighbors)

        curr_params_alpha_A_smaller, total_cost_alpha_A_smaller = self.optimize_jointly(self.X_bar_P, self.X_bar_A,
                                                                                             self.y_bar, self.d_X,
                                                                                             kappa_A, kappa_P,
                                                                                             alpha_A_smaller=True,
                                                                                             eps=eps,
                                                                                             show_convergence=show_convergence)
        curr_params_alpha_P_smaller, total_cost_alpha_P_smaller = self.optimize_jointly(self.X_bar_P, self.X_bar_A,
                                                                                             self.y_bar, self.d_X,
                                                                                             kappa_A, kappa_P,
                                                                                             alpha_A_smaller=False,
                                                                                             eps=eps,
                                                                                             show_convergence=show_convergence)

        if total_cost_alpha_A_smaller <= total_cost_alpha_P_smaller:
            return curr_params_alpha_A_smaller
        else:
            return curr_params_alpha_P_smaller

        # The first d_X coordinates are the same kinds of features shared between X_A and X_P
    def fit_momentum(self, X_A, X_P, y_P, kappa_A, kappa_P, r_A, r_P, eps=1e-5, num_nearest_neighbors=3,
                show_convergence=False):
            self.initialize(X_A, X_P, y_P, kappa_A, kappa_P, r_A, r_P, num_nearest_neighbors)

            # TODO: Can remove redundant code here
            curr_params_alpha_A_smaller, total_cost_alpha_A_smaller = self.optimize_jointly_with_momentum(self.X_bar_P, self.X_bar_A,
                                                                                            self.y_bar, self.d_X,
                                                                                            kappa_A, kappa_P,
                                                                                            alpha_A_smaller=True,
                                                                                            eps=eps,
                                                                                            show_convergence=show_convergence)
            curr_params_alpha_P_smaller, total_cost_alpha_P_smaller = self.optimize_jointly_with_momentum(self.X_bar_P, self.X_bar_A,
                                                                                            self.y_bar, self.d_X,
                                                                                            kappa_A, kappa_P,
                                                                                            alpha_A_smaller=False,
                                                                                            eps=eps,
                                                                                            show_convergence=show_convergence)

            return curr_params_alpha_A_smaller, total_cost_alpha_A_smaller, curr_params_alpha_P_smaller, total_cost_alpha_P_smaller

    def optimize_jointly_with_momentum(self, X_bar_P, X_bar_A, y_bar, d_X, kappa_A, kappa_P, alpha_A_smaller, eps=1e-5,
                         show_convergence=False):
        if alpha_A_smaller:
            cost_func = self.cost_with_alpha_A
            X_bar = X_bar_P
        else:
            cost_func = self.cost_with_alpha_P
            X_bar = X_bar_A

        _, d_X_plus_A = X_bar.shape
        theta_dim_first_half = d_X
        theta_dim_second_half = d_X_plus_A - theta_dim_first_half

        my_projector = projector.Projector(theta_dim_first_half, theta_dim_second_half, kappa_A, alpha_A_smaller)

        self.random_key, weights_random_key = random.split(self.random_key, 2)
        init_weights = random.normal(weights_random_key, (d_X_plus_A,))
        init_bias = 0.0
        alpha_A = 1.0
        alpha_P = 1.0

        curr_params_with_alphas = ParamsWithAlphas(init_weights, init_bias, alpha_A, alpha_P)
        prev_params_with_alphas = curr_params_with_alphas

        @jit
        def update(curr_params_with_alphas, prev_params_with_alphas, X_bar, y_bar, kappa_P, t):
            # Take derivative with respect to params
            point_for_grad = jax.tree_multimap(
                lambda curr_param, prev_param: curr_param + (t-2)/(t+1) * (curr_param - prev_param),
                curr_params_with_alphas, prev_params_with_alphas)

            curr_grad = grad(cost_func, argnums=0)(point_for_grad, X_bar, y_bar, kappa_P)

            new_params = jax.tree_multimap(
                lambda param, g: param - g * self.learning_rate, curr_params_with_alphas, curr_grad)

            return new_params

        def project(params_with_alphas):
            theta_1 = params_with_alphas.weights[:d_X]
            theta_2 = params_with_alphas.weights[d_X:]
            new_theta_1, new_theta_2, new_alpha_A, new_alpha_P = my_projector.project(theta_1, theta_2,
                                                                           params_with_alphas.alpha_A,
                                                                           params_with_alphas.alpha_P,
                                                                           eps)
            new_weights = jnp.concatenate([new_theta_1, new_theta_2])
            return ParamsWithAlphas(new_weights, params_with_alphas.bias, new_alpha_A, new_alpha_P)

        if show_convergence:
            cost_hist = []
            for curr_iter in range(self.n_iters):
                new_params_before_projection = update(curr_params_with_alphas, prev_params_with_alphas, X_bar, y_bar, kappa_P, curr_iter+1)
                prev_params_with_alphas = curr_params_with_alphas

                curr_params_before_projection = new_params_before_projection
                curr_params_with_alphas = project(curr_params_before_projection)

                if curr_iter % 20 == 0:
                    curr_cost = cost_func(curr_params_with_alphas, X_bar, y_bar, kappa_P)
                    cost_hist.append(curr_cost)

            fig = plt.figure()
            plt.plot(cost_hist)
            fig.suptitle('Convergence history')
            plt.xlabel('Iteration')
            plt.ylabel('Cost value')
            plt.show()
        else:
            for curr_iter in range(self.n_iters):
                curr_params_before_projection = update(curr_params_with_alphas, X_bar, y_bar, kappa_P)
                curr_params_with_alphas = project(curr_params_before_projection, kappa_A)

        return curr_params_with_alphas, cost_func(curr_params_with_alphas, X_bar, y_bar, kappa_P)

    def optimize_jointly(self, X_bar_P, X_bar_A, y_bar, d_X, kappa_A, kappa_P, alpha_A_smaller, eps=1e-5,
                         show_convergence=False):
        if alpha_A_smaller:
            cost_func = self.cost_with_alpha_A
            X_bar = X_bar_P
        else:
            cost_func = self.cost_with_alpha_P
            X_bar = X_bar_A

        _, d_X_plus_A = X_bar.shape
        theta_dim_first_half = d_X
        theta_dim_second_half = d_X_plus_A - theta_dim_first_half

        my_projector = projector.Projector(theta_dim_first_half, theta_dim_second_half, kappa_A, alpha_A_smaller)

        self.random_key, weights_random_key = random.split(self.random_key, 2)
        init_weights = random.normal(weights_random_key, (d_X_plus_A,))
        init_bias = 0.0
        alpha_A = 1.0
        alpha_P = 1.0

        curr_params_with_alphas = ParamsWithAlphas(init_weights, init_bias, alpha_A, alpha_P)

        @jit
        def update(params_with_alphas, X_bar, y_bar, kappa_P):
            # Take derivative with respect to params
            curr_grad = grad(cost_func, argnums=0)(params_with_alphas, X_bar, y_bar, kappa_P)

            new_params = jax.tree_multimap(
                lambda param, g: param - g * self.learning_rate, params_with_alphas, curr_grad)

            return new_params

        def project(params_with_alphas):
            theta_1 = params_with_alphas.weights[:d_X]
            theta_2 = params_with_alphas.weights[d_X:]
            new_theta_1, new_theta_2, new_alpha_A, new_alpha_P = my_projector.project(theta_1, theta_2,
                                                                           params_with_alphas.alpha_A,
                                                                           params_with_alphas.alpha_P,
                                                                           eps)
            new_weights = jnp.concatenate([new_theta_1, new_theta_2])
            return ParamsWithAlphas(new_weights, params_with_alphas.bias, new_alpha_A, new_alpha_P)

        if show_convergence:
            cost_hist = []
            for curr_iter in range(self.n_iters):
                curr_params_before_projection = update(curr_params_with_alphas, X_bar, y_bar, kappa_P)
                curr_params_with_alphas = project(curr_params_before_projection)

                if curr_iter % 20 == 0:
                    curr_cost = cost_func(curr_params_with_alphas, X_bar, y_bar, kappa_P)
                    cost_hist.append(curr_cost)

            fig = plt.figure()
            plt.plot(cost_hist)
            fig.suptitle('Convergence history')
            plt.xlabel('Iteration')
            plt.ylabel('Cost value')
            plt.show()
        else:
            for curr_iter in range(self.n_iters):
                curr_params_before_projection = update(curr_params_with_alphas, X_bar, y_bar, kappa_P)
                curr_params_with_alphas = project(curr_params_before_projection)

        return curr_params_with_alphas, cost_func(curr_params_with_alphas, X_bar, y_bar, kappa_P)

    def get_X_bar_y_bar_A(self, X_A, y_P):
        X_bar = np.array([X_A[i] for (i, j) in self.matchings])
        y_bar = np.array([y_P[j] for (i, j) in self.matchings])

        return jnp.array(X_bar), jnp.array(y_bar)

    def get_X_bar_y_bar_P(self, X_A, X_P, y_P, d_X):
        X_bar = np.array([np.concatenate([X_P[j, :d_X], X_A[i, d_X:]]) for (i, j) in self.matchings])
        y_bar = np.array([y_P[j] for (i, j) in self.matchings])

        return jnp.array(X_bar), jnp.array(y_bar)

    def get_pairwise_distance_avg(self, X_A, X_P, d_X):
        n_A, _ = X_A.shape
        n_P, _ = X_P.shape

        result = np.array(
            [np.linalg.norm(X_A[indx_pairs[0], :d_X][:d_X] - X_P[indx_pairs[1]], 2) for indx_pairs in self.matchings])
        return np.mean(result)

    ######## Reason for two cost functions ########
    # Just to avoid the headache with if statements with jax, we have two functions for cost
    # Also, you may think that why don't we just have another alpha input that multiplies the pairwise_distance_avg
    # However, this approach doesn't work because you want the autograd to be with respect to the same alpha that's in
    # params_with_alphas.
    @partial(jax.jit, static_argnums=(0,))
    def cost_with_alpha_A(self, params_with_alphas, X_bar, y_bar, kappa_P):
        n_bar, _ = X_bar.shape

        wX = jnp.dot(X_bar[:, ], params_with_alphas.weights) + params_with_alphas.bias
        ywX = jnp.multiply(y_bar, wX)

        logistic_loss = jnp.logaddexp(jnp.zeros(n_bar), -ywX)
        max_term = jnp.maximum(0, wX - params_with_alphas.alpha_P * kappa_P)

        unnormalized_cost = logistic_loss + max_term
        partial_cost = jnp.sum(unnormalized_cost) / n_bar

        total_loss = partial_cost + (params_with_alphas.alpha_A * self.r_A) + (params_with_alphas.alpha_P * self.r_P) \
                     - (params_with_alphas.alpha_A * self.pairwise_distance_avg)

        return total_loss

    @partial(jax.jit, static_argnums=(0,))
    def cost_with_alpha_P(self, params_with_alphas, X_bar, y_bar, kappa_P):
        n_bar, _ = X_bar.shape

        wX = jnp.dot(X_bar[:, ], params_with_alphas.weights) + params_with_alphas.bias
        ywX = jnp.multiply(y_bar, wX)

        logistic_loss = jnp.logaddexp(jnp.zeros(n_bar), -ywX)
        max_term = jnp.maximum(0, wX - params_with_alphas.alpha_P * kappa_P)

        unnormalized_cost = logistic_loss + max_term
        partial_cost = jnp.sum(unnormalized_cost) / n_bar

        total_loss = partial_cost + (params_with_alphas.alpha_A * self.r_A) + (params_with_alphas.alpha_P * self.r_P) \
                     - (params_with_alphas.alpha_P * self.pairwise_distance_avg)

        return total_loss

    def logistic(self, r):
        return 1 / (1 + jnp.exp(-r))

    def predict(self, X, params):
        return self.logistic(jnp.dot(X, params.weights) + params.bias)
