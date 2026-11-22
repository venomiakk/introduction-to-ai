import numpy as np

from experiments import DEFAULT_PARAMS, RESULTS_EXCEL, run_parameter_experiments
from plotter import PLOTS_DIR, plot_contour_swarm, plot_convergence
from pso_algorithm import ParticleSwarmOptimizer, objective_function


def calculate_best_value(bounds=(-4.5, 4.5)):
    """Calculates the global minimum of objective_function on bounds via closed-form x*(y) and 1D refinement."""
    low, high = bounds
    for _ in range(4):
        ys = np.linspace(low, high, 1000)
        a1, a2, a3 = 1.0 + ys, 1.0 - ys ** 2, 1.0 - ys ** 3
        xs = np.clip(
            (1.5 * a1 + 2.25 * a2 + 2.625 * a3) / (a1 ** 2 + a2 ** 2 + a3 ** 2),
            bounds[0],
            bounds[1],
        )
        vals = objective_function(xs, ys)
        idx = int(np.argmin(vals))
        step = (high - low) / 999
        low, high = max(bounds[0], ys[idx] - step), min(bounds[1], ys[idx] + step)
    return float(xs[idx]), float(ys[idx]), float(vals[idx])


def main():
    ref_x, ref_y, ref_val = calculate_best_value()
    print(f"Calculated Global Minimum: f({ref_x:.6f}, {ref_y:.6f}) = {ref_val:.8f}\n")

    print("=== Exercise 2: Particle Swarm Optimization (PSO) ===")
    print("Function: f(x, y) = (1.5 - x - xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2")
    print("Domain: x, y in [-4.5, 4.5]\n")

    print("=== Running Single Default PSO Solve ===")
    pso = ParticleSwarmOptimizer(**DEFAULT_PARAMS)
    result = pso.run(verbose=True)

    print(f"\nBest Solution Found: x = {result['best_x']:.6f}, y = {result['best_y']:.6f}")
    print(f"Minimum Objective Value f(x, y) = {result['best_val']:.8f}")
    print(f"Iterations Run: {result['iterations_run']} (Stopped early: {result['stopped_early']})")
    print(f"Execution Time: {result['execution_time_ms']:.3f} ms")

    plot_convergence(result, PLOTS_DIR / "best_run_convergence.png")
    plot_contour_swarm(result, PLOTS_DIR / "swarm_contour_landscape.png")
    print(f"Saved convergence and contour landscape plots to {PLOTS_DIR}")

    print("\n=== Running PSO Parameter Study Experiments ===")
    run_parameter_experiments()
    print(f"\nSaved experiment summary table to {RESULTS_EXCEL}")


if __name__ == "__main__":
    ref_x, ref_y, ref_val = calculate_best_value()
    print(f"Calculated Global Minimum: f({ref_x:.6f}, {ref_y:.6f}) = {ref_val:.8f}\n")

    print("=== Running Single Default PSO Solve ===")
    pso = ParticleSwarmOptimizer(**DEFAULT_PARAMS)
    result = pso.run(verbose=True)

    print(f"\nBest Solution Found: x = {result['best_x']:.6f}, y = {result['best_y']:.6f}")
    print(f"Minimum Objective Value f(x, y) = {result['best_val']:.8f}")
    print(f"Iterations Run: {result['iterations_run']} (Stopped early: {result['stopped_early']})")
    print(f"Execution Time: {result['execution_time_ms']:.3f} ms")
