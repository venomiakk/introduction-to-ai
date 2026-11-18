import random
import time


def objective_function(x, y):
    """
    Objective function from IAI_2.md:
    f(x, y) = (1.5 - x - x*y)^2 + (2.25 - x + x*y^2)^2 + (2.625 - x + x*y^3)^2
    for x, y in [-4.5, 4.5].
    """
    term1 = (1.5 - x - x * y) ** 2
    term2 = (2.25 - x + x * (y ** 2)) ** 2
    term3 = (2.625 - x + x * (y ** 3)) ** 2
    return term1 + term2 + term3


class ParticleSwarmOptimizer:
    def __init__(
        self,
        func=objective_function,
        bounds=(-4.5, 4.5),
        num_particles=30,
        iterations=100,
        w=0.7,
        c1=1.5,
        c2=1.5,
        max_no_improvement=30,
        tol=1e-9,
    ):
        self.func = func
        self.min_bound, self.max_bound = bounds
        self.num_particles = num_particles
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_no_improvement = max_no_improvement
        self.tol = tol

        # Max velocity limited to half the search range to prevent violent oscillation
        self.v_max = (self.max_bound - self.min_bound) * 0.5

    def init_swarm(self):
        span = self.max_bound - self.min_bound
        particles = []

        for _ in range(self.num_particles):
            x = random.uniform(self.min_bound, self.max_bound)
            y = random.uniform(self.min_bound, self.max_bound)
            vx = random.uniform(-span * 0.1, span * 0.1)
            vy = random.uniform(-span * 0.1, span * 0.1)
            val = self.func(x, y)

            particles.append({
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "val": val,
                "pbest_x": x,
                "pbest_y": y,
                "pbest_val": val,
            })
        return particles

    def clamp(self, val, low, high):
        return max(low, min(high, val))

    def run(self, verbose=False):
        start_clock = time.perf_counter()
        particles = self.init_swarm()

        initial_positions = [(p["x"], p["y"]) for p in particles]

        best_p = min(particles, key=lambda p: p["pbest_val"])
        gbest_x, gbest_y, gbest_val = best_p["pbest_x"], best_p["pbest_y"], best_p["pbest_val"]

        best_history = [gbest_val]
        avg_history = [sum(p["val"] for p in particles) / self.num_particles]
        gbest_path = [(gbest_x, gbest_y)]

        no_improvement = 0
        stopped_early = False
        iterations_run = 0

        for it in range(1, self.iterations + 1):
            iterations_run = it
            prev_gbest_val = gbest_val

            for p in particles:
                r1_x, r2_x = random.random(), random.random()
                r1_y, r2_y = random.random(), random.random()

                # Update velocity: inertia + cognitive + social
                p["vx"] = (
                    self.w * p["vx"]
                    + self.c1 * r1_x * (p["pbest_x"] - p["x"])
                    + self.c2 * r2_x * (gbest_x - p["x"])
                )
                p["vy"] = (
                    self.w * p["vy"]
                    + self.c1 * r1_y * (p["pbest_y"] - p["y"])
                    + self.c2 * r2_y * (gbest_y - p["y"])
                )

                p["vx"] = self.clamp(p["vx"], -self.v_max, self.v_max)
                p["vy"] = self.clamp(p["vy"], -self.v_max, self.v_max)

                # Update position and clamp to search domain [-4.5, 4.5]
                p["x"] = self.clamp(p["x"] + p["vx"], self.min_bound, self.max_bound)
                p["y"] = self.clamp(p["y"] + p["vy"], self.min_bound, self.max_bound)

                # Evaluate objective function
                p["val"] = self.func(p["x"], p["y"])

                # Update personal best
                if p["val"] < p["pbest_val"]:
                    p["pbest_x"] = p["x"]
                    p["pbest_y"] = p["y"]
                    p["pbest_val"] = p["val"]

                    # Update global best
                    if p["val"] < gbest_val:
                        gbest_x = p["x"]
                        gbest_y = p["y"]
                        gbest_val = p["val"]

            if prev_gbest_val - gbest_val > self.tol:
                no_improvement = 0
            else:
                no_improvement += 1

            avg_val = sum(p["val"] for p in particles) / self.num_particles
            best_history.append(gbest_val)
            avg_history.append(avg_val)
            gbest_path.append((gbest_x, gbest_y))

            if verbose and (it % 10 == 0 or it == 1 or it == self.iterations):
                print(
                    f"[Iter {it:3d}/{self.iterations}] "
                    f"Best f(x,y): {gbest_val:.6f} at ({gbest_x:.5f}, {gbest_y:.5f}) | "
                    f"No Improv: {no_improvement}"
                )

            if self.max_no_improvement and no_improvement >= self.max_no_improvement:
                stopped_early = True
                if verbose:
                    print(f"Early stopping at iteration {it}.")
                break

        final_positions = [(p["x"], p["y"]) for p in particles]

        return {
            "best_x": gbest_x,
            "best_y": gbest_y,
            "best_val": gbest_val,
            "best_history": best_history,
            "avg_history": avg_history,
            "gbest_path": gbest_path,
            "initial_positions": initial_positions,
            "final_positions": final_positions,
            "iterations_run": iterations_run,
            "stopped_early": stopped_early,
            "execution_time_ms": (time.perf_counter() - start_clock) * 1000.0,
        }
