import random
import time


class JobSchedulingGA:
    def __init__(
        self,
        tasks,
        pop_size=100,
        generations=300,
        crossover_prob=0.85,
        mutation_prob=0.15,
        crossover_type="pox",
        mutation_type="swap",
        selection_type="tournament",
        tournament_size=3,
        elitism_count=2,
        max_no_improvement=60,
    ):
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.ops_per_task = [len(task) for task in tasks]
        self.max_resource_id = max(res for task in tasks for res, _ in task)

        self.pop_size = pop_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.selection_type = selection_type
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.max_no_improvement = max_no_improvement

        # Template chromosome: each task_id repeated len(tasks[task_id]) times
        self.base_genes = [
            task_id for task_id, task in enumerate(tasks) for _ in range(len(task))
        ]

    def init_population(self):
        population = []
        for _ in range(self.pop_size):
            chromosome = self.base_genes.copy()
            random.shuffle(chromosome)
            population.append(chromosome)
        return population

    def evaluate_makespan(self, chromosome):
        task_op_idx = [0] * self.num_tasks
        task_ready = [0] * self.num_tasks
        res_ready = [0] * (self.max_resource_id + 1)

        for task_id in chromosome:
            op_idx = task_op_idx[task_id]
            task_op_idx[task_id] = op_idx + 1

            resource, duration = self.tasks[task_id][op_idx]
            start_time = max(task_ready[task_id], res_ready[resource])
            end_time = start_time + duration

            task_ready[task_id] = end_time
            res_ready[resource] = end_time

        return max(task_ready)

    def decode_schedule(self, chromosome):
        """Returns a list of dicts with start/end times of each operation for the Gantt chart."""
        task_op_idx = [0] * self.num_tasks
        task_ready = [0] * self.num_tasks
        res_ready = [0] * (self.max_resource_id + 1)

        schedule = []
        for task_id in chromosome:
            op_idx = task_op_idx[task_id]
            task_op_idx[task_id] = op_idx + 1

            resource, duration = self.tasks[task_id][op_idx]
            start_time = max(task_ready[task_id], res_ready[resource])
            end_time = start_time + duration

            task_ready[task_id] = end_time
            res_ready[resource] = end_time

            schedule.append({
                "task_id": task_id,
                "resource": resource,
                "duration": duration,
                "start_time": start_time,
                "end_time": end_time,
            })
        return schedule

    def select_parent(self, population, makespans, roulette_weights=None):
        if self.selection_type == "tournament":
            candidates = random.sample(range(self.pop_size), self.tournament_size)
            best_idx = min(candidates, key=lambda i: makespans[i])
            return population[best_idx]
        else:  # roulette
            return random.choices(population, weights=roulette_weights, k=1)[0]

    def crossover_pox(self, parent1, parent2):
        """Precedence Preservative Crossover (POX)."""
        group_a = set(random.sample(range(self.num_tasks), self.num_tasks // 2))

        def make_child(p1, p2):
            child = [gene if gene in group_a else -1 for gene in p1]
            p2_genes = (gene for gene in p2 if gene not in group_a)
            for i in range(len(child)):
                if child[i] == -1:
                    child[i] = next(p2_genes)
            return child

        return make_child(parent1, parent2), make_child(parent2, parent1)

    def crossover_ox(self, parent1, parent2):
        """Generalized Order Crossover (OX)."""
        start, end = sorted(random.sample(range(len(parent1)), 2))

        def make_child(p1, p2):
            child = [-1] * len(p1)
            needed = self.ops_per_task.copy()
            for i in range(start, end):
                child[i] = p1[i]
                needed[p1[i]] -= 1

            pos = 0
            for gene in p2:
                if needed[gene] > 0:
                    if pos == start:
                        pos = end
                    child[pos] = gene
                    needed[gene] -= 1
                    pos += 1
            return child

        return make_child(parent1, parent2), make_child(parent2, parent1)

    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        if self.crossover_type == "ox":
            return self.crossover_ox(parent1, parent2)
        return self.crossover_pox(parent1, parent2)

    def mutate(self, chromosome):
        if random.random() <= self.mutation_prob:
            i, j = sorted(random.sample(range(len(chromosome)), 2))
            if self.mutation_type == "swap":
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
            elif self.mutation_type == "inversion":
                chromosome[i : j + 1] = reversed(chromosome[i : j + 1])

    def run(self, verbose=False):
        start_clock = time.perf_counter()

        population = self.init_population()
        makespans = [self.evaluate_makespan(ind) for ind in population]

        best_idx = min(range(self.pop_size), key=lambda i: makespans[i])
        best_makespan = makespans[best_idx]
        best_chromosome = population[best_idx].copy()

        best_history = [best_makespan]
        avg_history = [sum(makespans) / self.pop_size]

        no_improvement = 0
        stopped_early = False
        generations_run = 0

        for gen in range(1, self.generations + 1):
            generations_run = gen
            ranked_indices = sorted(range(self.pop_size), key=lambda i: makespans[i])
            new_population = [
                population[i].copy() for i in ranked_indices[: self.elitism_count]
            ]

            roulette_weights = None
            if self.selection_type == "roulette":
                max_c = max(makespans)
                roulette_weights = [(max_c - c + 1.0) / c for c in makespans]

            while len(new_population) < self.pop_size:
                p1 = self.select_parent(population, makespans, roulette_weights)
                p2 = self.select_parent(population, makespans, roulette_weights)

                c1, c2 = self.crossover(p1, p2)
                self.mutate(c1)
                new_population.append(c1)

                if len(new_population) < self.pop_size:
                    self.mutate(c2)
                    new_population.append(c2)

            population = new_population
            makespans = [self.evaluate_makespan(ind) for ind in population]

            gen_best_idx = min(range(self.pop_size), key=lambda i: makespans[i])
            gen_best = makespans[gen_best_idx]
            gen_avg = sum(makespans) / self.pop_size

            if gen_best < best_makespan:
                best_makespan = gen_best
                best_chromosome = population[gen_best_idx].copy()
                no_improvement = 0
            else:
                no_improvement += 1

            best_history.append(best_makespan)
            avg_history.append(gen_avg)

            if verbose and (gen % 25 == 0 or gen == 1 or gen == self.generations):
                print(
                    f"[Gen {gen:4d}/{self.generations}] "
                    f"Best: {best_makespan} | Avg: {gen_avg:.1f} | No Improv: {no_improvement}"
                )

            if self.max_no_improvement and no_improvement >= self.max_no_improvement:
                stopped_early = True
                if verbose:
                    print(f"Early stopping at generation {gen}.")
                break

        return {
            "best_chromosome": best_chromosome,
            "best_makespan": best_makespan,
            "best_history": best_history,
            "avg_history": avg_history,
            "generations_run": generations_run,
            "stopped_early": stopped_early,
            "execution_time_s": time.perf_counter() - start_clock,
            "schedule": self.decode_schedule(best_chromosome),
        }
