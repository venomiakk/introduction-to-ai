import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from pso_algorithm import objective_function


PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def plot_convergence(result, output_path, title="PSO Convergence Curve"):
    """Plots best objective value across iterations (log scale)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = range(len(result["best_history"]))
    ax.plot(
        iterations,
        result["best_history"],
        label=f"Global Best f(x, y) (Final = {result['best_val']:.6f})",
        linewidth=2.0,
    )

    ax.set_yscale("log")
    ax.set_title(
        f"{title}\nBest: ({result['best_x']:.5f}, {result['best_y']:.5f}), "
        f"f(x, y) = {result['best_val']:.6f}"
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value f(x, y) [Log Scale]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_contour_swarm(result, output_path, bounds=(-4.5, 4.5)):
    """Plots a 2D contour map of f(x, y) showing initial/final swarm positions and global best."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))

    low, high = bounds
    grid_x = np.linspace(low, high, 300)
    grid_y = np.linspace(low, high, 300)
    X, Y = np.meshgrid(grid_x, grid_y)
    Z = objective_function(X, Y)

    # Use log-spaced contour levels because f(x,y) ranges from ~0.246 up to >10^5
    levels = np.logspace(np.log10(max(Z.min(), 1e-2)), np.log10(Z.max()), 35)
    contour = ax.contourf(X, Y, Z, levels=levels, norm=LogNorm(), cmap="viridis", alpha=0.85)
    fig.colorbar(contour, ax=ax, label="f(x, y) [Log Scale]")

    init_x = [pos[0] for pos in result["initial_positions"]]
    init_y = [pos[1] for pos in result["initial_positions"]]
    final_x = [pos[0] for pos in result["final_positions"]]
    final_y = [pos[1] for pos in result["final_positions"]]
    path_x = [pos[0] for pos in result["gbest_path"]]
    path_y = [pos[1] for pos in result["gbest_path"]]

    ax.scatter(
        init_x, init_y, color="white", edgecolor="black", s=45, alpha=0.8, label="Initial Particles"
    )
    ax.plot(
        path_x, path_y, color="orange", linestyle="--", linewidth=1.8, label="Global Best Trajectory"
    )
    ax.scatter(
        final_x, final_y, color="cyan", edgecolor="black", s=50, alpha=0.9, label="Final Particles"
    )
    ax.scatter(
        [result["best_x"]],
        [result["best_y"]],
        color="red",
        marker="*",
        s=220,
        edgecolor="black",
        zorder=5,
        label=f"Global Best ({result['best_x']:.4f}, {result['best_y']:.4f})",
    )

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"PSO Swarm on Objective Function Landscape (Best f = {result['best_val']:.6f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_parameter_comparison(group_title, curves, num_runs, output_path):
    """Plots average convergence curves for different values of a tested PSO parameter."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, mean_curve in curves:
        ax.plot(range(len(mean_curve)), mean_curve, linewidth=1.8, label=label)

    ax.set_yscale("log")
    ax.set_title(f"Effect of {group_title} on Convergence (Averaged over {num_runs} Runs)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best f(x, y) [Log Scale]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
