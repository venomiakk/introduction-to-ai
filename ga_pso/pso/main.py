from experiments import DEFAULT_PARAMS, RESULTS_EXCEL, run_parameter_experiments
from plotter import PLOTS_DIR, plot_contour_swarm, plot_convergence
from pso_algorithm import ParticleSwarmOptimizer


def main():
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
    main()
