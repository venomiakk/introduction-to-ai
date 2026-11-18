from pathlib import Path
from openpyxl import Workbook

from genetic_algorithm import JobSchedulingGA
from plotter import PLOTS_DIR, plot_parameter_comparison


RESULTS_EXCEL = Path(__file__).resolve().parent / "ga_experiment_results.xlsx"

DEFAULT_PARAMS = {
    "pop_size": 100,
    "generations": 300,
    "crossover_prob": 0.85,
    "mutation_prob": 0.15,
    "crossover_type": "pox",
    "mutation_type": "swap",
    "selection_type": "tournament",
    "tournament_size": 3,
    "elitism_count": 2,
    "max_no_improvement": 60,
}


def save_results_to_excel(records, output_file=RESULTS_EXCEL):
    wb = Workbook()
    ws = wb.active
    ws.title = "GA_Experiments"

    ws.append([
        "Experiment Group",
        "Parameter",
        "Value",
        "Best Makespan",
        "Avg Makespan",
        "Worst Makespan",
        "Avg Generations Run",
        "Early Stops",
        "Avg Execution Time (s)",
    ])

    for rec in records:
        ws.append([
            rec["group"],
            rec["parameter"],
            rec["value"],
            rec["best_makespan"],
            rec["avg_makespan"],
            rec["worst_makespan"],
            rec["avg_generations"],
            rec["early_stops"],
            rec["avg_time_s"],
        ])

    wb.save(output_file)


def run_parameter_experiments(tasks, lower_bound, num_runs=5):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    experiment_groups = {
        "Population Size": ("pop_size", [30, 50, 100, 200]),
        "Generations Limit": ("generations", [50, 100, 200, 400]),
        "Crossover Probability": ("crossover_prob", [0.5, 0.7, 0.85, 0.95]),
        "Mutation Probability": ("mutation_prob", [0.02, 0.10, 0.20, 0.40]),
        "Crossover Operator": ("crossover_type", ["pox", "ox"]),
        "Mutation Operator": ("mutation_type", ["swap", "inversion"]),
        "Selection Method": ("selection_type", ["tournament", "roulette"]),
        "Tournament Size": ("tournament_size", [2, 3, 5, 7]),
        "Early Stopping Patience": ("max_no_improvement", [25, 50, 100, None]),
    }

    all_records = []

    for group_title, (param_name, values) in experiment_groups.items():
        print(f"\n--- Testing {group_title} ({param_name}) ---")
        curves = []

        for val in values:
            params = DEFAULT_PARAMS.copy()
            params[param_name] = val

            run_results = [JobSchedulingGA(tasks, **params).run() for _ in range(num_runs)]

            best_makespans = [r["best_makespan"] for r in run_results]
            exec_times = [r["execution_time_s"] for r in run_results]
            gens_used = [r["generations_run"] for r in run_results]
            early_stops = sum(1 for r in run_results if r["stopped_early"])

            min_c = min(best_makespans)
            avg_c = sum(best_makespans) / num_runs
            max_c = max(best_makespans)
            avg_time = sum(exec_times) / num_runs
            avg_gens = sum(gens_used) / num_runs

            label_val = "None (disabled)" if val is None else str(val)
            print(
                f"  {param_name}={label_val:<15} | "
                f"Best: {min_c} | Avg: {avg_c:.1f} | Max: {max_c} | "
                f"Avg Gens: {avg_gens:.1f} | Avg Time: {avg_time:.2f}s"
            )

            all_records.append({
                "group": group_title,
                "parameter": param_name,
                "value": label_val,
                "best_makespan": min_c,
                "avg_makespan": round(avg_c, 2),
                "worst_makespan": max_c,
                "avg_generations": round(avg_gens, 1),
                "early_stops": f"{early_stops}/{num_runs}",
                "avg_time_s": round(avg_time, 3),
            })

            max_len = max(len(r["best_history"]) for r in run_results)
            mean_curve = [
                sum(
                    r["best_history"][g] if g < len(r["best_history"]) else r["best_history"][-1]
                    for r in run_results
                ) / num_runs
                for g in range(max_len)
            ]
            curves.append((f"{param_name}={label_val} (avg={avg_c:.1f}, min={min_c})", mean_curve))

        plot_parameter_comparison(
            group_title=group_title,
            curves=curves,
            lower_bound=lower_bound,
            num_runs=num_runs,
            output_path=PLOTS_DIR / f"param_{param_name}.png",
        )

    save_results_to_excel(all_records)
    return all_records
