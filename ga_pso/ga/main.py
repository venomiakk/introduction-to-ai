from data_loader import DEFAULT_EXCEL_PATH, get_problem_summary, load_tasks_from_excel
from experiments import DEFAULT_PARAMS, RESULTS_EXCEL, run_parameter_experiments
from genetic_algorithm import JobSchedulingGA
from plotter import PLOTS_DIR, plot_convergence, plot_gantt_chart


def run_experiments():
    tasks = load_tasks_from_excel(DEFAULT_EXCEL_PATH)
    summary = get_problem_summary(tasks)
    lower_bound = summary["theoretical_lower_bound"]

    print("=== Problem Summary ===")
    print(f"Tasks: {summary['num_tasks']}, Operations/Task: {summary['num_ops_per_task']}")
    print(f"Total Operations: {summary['total_operations']}, Resources: {summary['num_resources']}")
    print(f"Theoretical Lower Bound: {lower_bound}")

    print("\n=== Running Single Default GA Solve ===")
    ga = JobSchedulingGA(tasks, **DEFAULT_PARAMS)
    result = ga.run(verbose=True)

    print(f"\nBest Makespan Found: {result['best_makespan']} (Lower bound: {lower_bound})")
    print(f"Generations Run: {result['generations_run']} (Stopped early: {result['stopped_early']})")
    print(f"Execution Time: {result['execution_time_s']:.3f} s")

    plot_convergence(result, lower_bound, PLOTS_DIR / "best_run_convergence.png")
    plot_gantt_chart(result, PLOTS_DIR / "best_schedule_gantt.png", num_tasks=len(tasks))
    print(f"Saved convergence and Gantt plots to {PLOTS_DIR}")

    print("\n=== Running Parameter Study Experiments ===")
    run_parameter_experiments(tasks, lower_bound=lower_bound)
    print(f"\nSaved experiment summary table to {RESULTS_EXCEL}")

def main():
    tasks = load_tasks_from_excel(DEFAULT_EXCEL_PATH)
    summary = get_problem_summary(tasks)
    lower_bound = summary["theoretical_lower_bound"]

    print("=== Problem Summary ===")
    print(f"Tasks: {summary['num_tasks']}, Operations/Task: {summary['num_ops_per_task']}")
    print(f"Total Operations: {summary['total_operations']}, Resources: {summary['num_resources']}")
    print(f"Theoretical Lower Bound: {lower_bound}")

    print("\n=== Running Single Default GA Solve ===")
    ga = JobSchedulingGA(tasks, **DEFAULT_PARAMS)
    result = ga.run(verbose=True)
    print(f"\nBest Makespan Found: {result['best_makespan']} (Lower bound: {lower_bound})")
    print(f"Generations Run: {result['generations_run']} (Stopped early: {result['stopped_early']})")
    print(f"Execution Time: {result['execution_time_s']:.3f} s")


if __name__ == "__main__":
    main()
