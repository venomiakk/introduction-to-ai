from pathlib import Path
from openpyxl import Workbook

from pso_algorithm import ParticleSwarmOptimizer
from plotter import PLOTS_DIR, plot_parameter_comparison


RESULTS_EXCEL = Path(__file__).resolve().parent / "pso_experiment_results.xlsx"

DEFAULT_PARAMS = {
    "num_particles": 30,
    "iterations": 100,
    "w": 0.7,
    "c1": 1.5,
    "c2": 1.5,
    "max_no_improvement": 30,
}


def save_results_to_excel(records, output_file=RESULTS_EXCEL):
    wb = Workbook()
    ws = wb.active
    ws.title = "PSO_Experiments"

    ws.append([
        "Experiment Group",
        "Parameter",
        "Value",
        "Best f(x, y)",
        "Avg f(x, y)",
        "Worst f(x, y)",
        "Best (x, y)",
        "Avg Iterations Run",
        "Early Stops",
        "Avg Execution Time (ms)",
    ])

    for rec in records:
        ws.append([
            rec["group"],
            rec["parameter"],
            rec["value"],
            rec["best_val"],
            rec["avg_val"],
            rec["worst_val"],
            rec["best_xy"],
            rec["avg_iterations"],
            rec["early_stops"],
            rec["avg_time_ms"],
        ])

    wb.save(output_file)


def run_parameter_experiments(num_runs=10):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    experiment_groups = {
        "Swarm Size (num_particles)": ("num_particles", [10, 20, 30, 50, 100]),
        "Iterations Limit": ("iterations", [20, 50, 100, 200]),
        "Inertia Weight (w)": ("w", [0.2, 0.5, 0.7, 0.9, 1.1]),
        "Cognitive Coefficient (c1)": ("c1", [0.5, 1.0, 1.5, 2.0, 2.5]),
        "Social Coefficient (c2)": ("c2", [0.5, 1.0, 1.5, 2.0, 2.5]),
        "Early Stopping Patience": ("max_no_improvement", [10, 20, 30, 50, None]),
    }

    all_records = []

    for group_title, (param_name, values) in experiment_groups.items():
        print(f"\n--- Testing {group_title} ({param_name}) ---")
        curves = []

        for val in values:
            params = DEFAULT_PARAMS.copy()
            params[param_name] = val

            run_results = [ParticleSwarmOptimizer(**params).run() for _ in range(num_runs)]

            best_vals = [r["best_val"] for r in run_results]
            exec_times = [r["execution_time_ms"] for r in run_results]
            iters_used = [r["iterations_run"] for r in run_results]
            early_stops = sum(1 for r in run_results if r["stopped_early"])

            best_run = min(run_results, key=lambda r: r["best_val"])
            min_f = best_run["best_val"]
            avg_f = sum(best_vals) / num_runs
            max_f = max(best_vals)
            avg_time = sum(exec_times) / num_runs
            avg_iters = sum(iters_used) / num_runs
            best_xy_str = f"({best_run['best_x']:.5f}, {best_run['best_y']:.5f})"

            label_val = "None (disabled)" if val is None else str(val)
            print(
                f"  {param_name}={label_val:<15} | "
                f"Best: {min_f:.6f} | Avg: {avg_f:.6f} | Worst: {max_f:.6f} | "
                f"Best (x,y): {best_xy_str} | Avg Iters: {avg_iters:.1f}"
            )

            all_records.append({
                "group": group_title,
                "parameter": param_name,
                "value": label_val,
                "best_val": round(min_f, 8),
                "avg_val": round(avg_f, 8),
                "worst_val": round(max_f, 8),
                "best_xy": best_xy_str,
                "avg_iterations": round(avg_iters, 1),
                "early_stops": f"{early_stops}/{num_runs}",
                "avg_time_ms": round(avg_time, 3),
            })

            max_len = max(len(r["best_history"]) for r in run_results)
            mean_curve = [
                sum(
                    r["best_history"][i] if i < len(r["best_history"]) else r["best_history"][-1]
                    for r in run_results
                ) / num_runs
                for i in range(max_len)
            ]
            curves.append((f"{param_name}={label_val} (avg={avg_f:.6f})", mean_curve))

        plot_parameter_comparison(
            group_title=group_title,
            curves=curves,
            num_runs=num_runs,
            output_path=PLOTS_DIR / f"param_{param_name}.png",
        )

    save_results_to_excel(all_records)
    return all_records
