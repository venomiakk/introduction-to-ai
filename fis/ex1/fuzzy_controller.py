import os
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class BrakingFuzzyController:
    """Mamdani Fuzzy Inference System (FIS) for Exercise 1: (distance, velocity) -> target_cmd."""

    def __init__(self, max_dist: float = 10.0, max_vel: float = 1.2):
        self.max_dist = max_dist
        self.max_vel = max_vel

        self.distance, self.velocity, self.target_cmd = self._define_variables()
        self.rules = self._define_rules()
        self.system = ctrl.ControlSystem(self.rules)
        self.sim = ctrl.ControlSystemSimulation(self.system)

    def _define_variables(self):
        distance = ctrl.Antecedent(np.arange(0.0, self.max_dist + 0.05, 0.05), "distance")
        velocity = ctrl.Antecedent(np.arange(0.0, self.max_vel + 0.02, 0.02), "velocity")
        target_cmd = ctrl.Consequent(np.arange(-1.0, 10.05, 0.1), "target_cmd")

        # Membership functions for distance to the wall [m]
        distance["very_close"] = fuzz.trapmf(distance.universe, [0.0, 0.0, 0.5, 1.2])
        distance["close"] = fuzz.trimf(distance.universe, [0.6, 1.8, 3.0])
        distance["medium"] = fuzz.trimf(distance.universe, [2.2, 4.0, 5.8])
        distance["far"] = fuzz.trapmf(distance.universe, [4.8, 6.2, self.max_dist, self.max_dist])

        # Membership functions for linear velocity [m/s]
        velocity["slow"] = fuzz.trapmf(velocity.universe, [0.0, 0.0, 0.15, 0.35])
        velocity["medium"] = fuzz.trimf(velocity.universe, [0.20, 0.45, 0.70])
        velocity["fast"] = fuzz.trapmf(velocity.universe, [0.55, 0.75, self.max_vel, self.max_vel])

        # Membership functions for target drive command passed to tank.forward(cmd)
        target_cmd["stop"] = fuzz.trimf(target_cmd.universe, [-1.0, 0.0, 1.0])
        target_cmd["creep"] = fuzz.trimf(target_cmd.universe, [0.5, 2.0, 4.0])
        target_cmd["medium"] = fuzz.trimf(target_cmd.universe, [3.0, 5.5, 8.0])
        target_cmd["fast"] = fuzz.trapmf(target_cmd.universe, [7.0, 8.5, 10.0, 10.0])

        return distance, velocity, target_cmd

    def _define_rules(self):
        d, v, out = self.distance, self.velocity, self.target_cmd
        return [
            ctrl.Rule(d["far"] & v["slow"], out["fast"]),
            ctrl.Rule(d["far"] & v["medium"], out["fast"]),
            ctrl.Rule(d["far"] & v["fast"], out["fast"]),

            ctrl.Rule(d["medium"] & v["slow"], out["medium"]),
            ctrl.Rule(d["medium"] & v["medium"], out["medium"]),
            ctrl.Rule(d["medium"] & v["fast"], out["creep"]),

            ctrl.Rule(d["close"] & v["slow"], out["creep"]),
            ctrl.Rule(d["close"] & v["medium"], out["stop"]),
            ctrl.Rule(d["close"] & v["fast"], out["stop"]),

            ctrl.Rule(d["very_close"] & v["slow"], out["stop"]),
            ctrl.Rule(d["very_close"] & v["medium"], out["stop"]),
            ctrl.Rule(d["very_close"] & v["fast"], out["stop"]),
        ]

    def compute(self, distance: float, velocity: float) -> float:
        """Compute target drive command from distance [m] and linear velocity [m/s]."""
        self.sim.input["distance"] = float(np.clip(distance, 0.0, self.max_dist))
        self.sim.input["velocity"] = float(np.clip(velocity, 0.0, self.max_vel))
        self.sim.compute()
        return float(self.sim.output["target_cmd"])

    def plot(self, save_path: str | None = None, show: bool = True):
        """Plot membership functions for all controller variables."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(9, 8))
        specs = [
            (axes[0], self.distance, "Distance to Wall (distance) [m]"),
            (axes[1], self.velocity, "Linear Velocity (velocity) [m/s]"),
            (axes[2], self.target_cmd, "Target Drive Command (target_cmd)"),
        ]

        for ax, var, title in specs:
            for term_name, term in var.terms.items():
                ax.plot(var.universe, term.mf, linewidth=2, label=term_name)
            ax.set_title(title)
            ax.set_ylabel("Membership")
            ax.set_ylim(-0.05, 1.08)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved membership function plot to: {save_path}")
        if show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    ex1_dir = os.path.dirname(os.path.abspath(__file__))
    controller = BrakingFuzzyController()
    controller.plot(save_path=os.path.join(ex1_dir, "membership_functions.png"), show=True)
