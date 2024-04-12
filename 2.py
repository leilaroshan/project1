import jax.numpy as jnp
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import colorsys
import matplotlib.patches as patches
from matplotlib import colors as mcolors
from skopt import gp_minimize
from skopt.plots import plot_gaussian_process
import argparse

# Bayesian Loss Call set to 30 Iterations


class BayesianOptimization:
    """Performs Bayesian optimization on a given objective function.

    Attributes:
        objective_func (callable): The objective function to optimize.
        space (list): The search space for optimization.

    Methods:
        optimize: Runs Bayesian optimization using gp_minimize
        plot_loss: Plots the loss over iterations
    """

    def __init__(self, objective_func, space):
        self.objective_func = objective_func
        self.space = space

    def optimize(self, n_calls=30, random_state=42):
        # Call gp_minimize with the objective function and search space
        result = gp_minimize(
            self.objective_func,
            self.space,
            acq_func="LCB",
            n_calls=n_calls,
            random_state=random_state,
        )

        # Print the optimized parameters and loss

        # Print the optimized parameters and loss
        print("Optimized parameters: ", result.x)
        print("Loss: ", result.fun)
        self.plot_loss(result)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = plot_gaussian_process(result, ax=ax)
        ax.set_xlabel("optimized offset (m)", fontsize=12)
        ax.set_ylabel("objective function", fontsize=12)
        ax.legend(loc="upper left", fontsize=12)
        ax.set_title("bayesian uncertainty over optimized offset", fontsize=12)
        plt.savefig("bayesian_optimization_gaussian_process.png", dpi=200)

        # Save the loss data to plot
        # self.save_plot_data(result, "bayesian_optimization_loss_data.csv")

        # # Save uncertainty data
        # X_train = result.x_iters  # Training data for x (offset optimized)
        # y_train = result.func_vals  # Training data for y (objective function)
        # x_test = np.linspace(self.space[0][0], self.space[0][1], 100)  # Test data for x
        # mu_s, cov_s = self.omega_opt(X_train, y_train, x_test, result.x)
        # self.save_uncertainty_data(X_train, y_train, x_test, mu_s, cov_s, "bayesian_optimization_uncertainty_data")

        return result.x

    # # Saving the data to plot
    # Saving the data to plot
    # def save_plot_data(self, result, filename):
    #     x_iters = result.x_iters
    #     y_vals = result.func_vals
    #     data = np.column_stack((x_iters, y_vals))
    #     np.savetxt(filename, data, header="x_iters, func_vals", delimiter=",")

    # def save_uncertainty_data(self, X_train, y_train, x_test, mu_s, cov_s, filename):
    #     data = np.column_stack((X_train, y_train))
    #     np.savetxt(filename + "_train_data.txt", data, header="X_train, y_train", delimiter=",")
    #     np.savetxt(filename + "_test_data.txt", x_test, header="x_test", delimiter=",")
    #     np.savetxt(filename + "_mu_s.txt", mu_s, header="mu_s", delimiter=",")
    #     np.savetxt(filename + "_cov_s.txt", cov_s, header="cov_s", delimiter=",")

    # Plots
    def plot_loss(self, result):
        x_iters = result.x_iters
        y_vals = result.func_vals
        iters = range(1, len(x_iters) + 1)
        improved_i = [1]
        improved_y = [y_vals[0]]
        for i, y_val in zip(iters, y_vals):
            if y_val <= improved_y[-1]:
                improved_i.append(i)
                improved_y.append(y_val)
        # print(improved_i, improved_y)
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plotting the loss over iterations
        ax.plot(
            iters,
            y_vals,
            marker="o",
            linestyle="",
            color="b",
            label="Objective Function",
        )
        ax.plot(
            improved_i,
            improved_y,
            marker="o",
            linestyle="-",
            color="r",
            label="Minimum",
        )
        ax.set_xlabel("Iterations", fontsize=12)
        ax.set_ylabel("Objective Function", fontsize=12)
        ax.legend(loc="upper right", fontsize=12)
        ax.grid(True)
        # Set y-axis to log scale
        ax.set_yscale("log")

        # ax.set_ylim(1e-3, 1)  # Set limits from 10^-3 to 0
        ax.set_title("Optimization Loss over Iterations", fontsize=12)

        plt.title("Optimization Loss over Iterations")
        plt.savefig("bayesian_optimization_loss.png")

    # save x and y data of plot
    # self.save_plot_data(result, "bayesian_optimization_loss_data.csv")

    # Uncertainty in Bayesian Optimization
    # Define the Radial Basis Function kernel
    def rbf_kernel(x1, x2, l=1.0, sigma_f=1.0):
        sqdist = (
            np.sum(x1**2, 1).reshape(-1, 1)
            + np.sum(x2**2, 1)
            - 2 * np.dot(x1, x2.T)
        )
        return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

    def omega_opt(self, X_train, y_train, x_test, target, l=30, sigma_f=0.5):
        # Generating the x and y data arrays with floats
        X_train = X_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)
        # Compute the posterior mean and covariance
        K = self.rbf_kernel(X_train, X_train, l=l, sigma_f=sigma_f)
        K_s = self.rbf_kernel(X_train, x_test, l=l, sigma_f=sigma_f)
        K_ss = (
            self.rbf_kernel(x_test, x_test, l=l, sigma_f=sigma_f)
            + np.identity(x_test.shape[0]) * 1e-5
        )  # adding uncertainty to covariance matrix of testing data
        K_inv = np.linalg.inv(K)

        # Computation for mu_s and cov_s:
        mu_s = K_s.T.dot(K_inv).dot(y_train)
        mu_mean = mu_s.mean()  # Calculate the mean
        print("Mean of GP posterior mean (mu_s):", mu_mean)  # Print the mean
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        # Sampling several functions from the posterior
        num_samples = 10
        f_post = np.random.multivariate_normal(mu_s, cov_s, num_samples)

        # Plot the data, the mean, and 95% confidence interval of the posterior
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train, y_train, c="red", s=50, zorder=10, edgecolors=(0, 0, 0))
        plt.plot(x_test, mu_s, "k", lw=1, zorder=9)
        plt.fill_between(
            x_test.flatten(),
            mu_s - 1.96 * np.sqrt(np.diag(cov_s)),
            mu_s + 1.96 * np.sqrt(np.diag(cov_s)),
            color="blue",
            alpha=0.2,
        )

        # for i in range(num_samples):
        #     plt.plot(x_test, f_post[i], lw=0.5)

        plt.tight_layout()
        plt.savefig("bayesian_optimization_uncertainty.png", dpi=200)
        plt.show()


# different l and sigma_f values
# for lscale in [5, 10, 15, 20, 25, 30]:
#     omega_opt(sigma_f=1.0, l=lscale)

# for amplitude in [0.1, 1.0, 2.0, 8.0]:
#     omega_opt(sigma_f=amplitude, l=30.0)
# Save the plot data
# self.save_uncertainty_data(X_train, y_train, x_test, mu_s, cov_s, "bayesian_optimization_uncertainty_data.csv")
