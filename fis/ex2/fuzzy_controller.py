import os
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class BayEntryFuzzyController:
    """
    Mamdani Fuzzy Inference System for entering and centering inside a parking bay
    between two flanking cars using distance sensors only.

    Inputs:
        - side_diff: left_dist - right_dist [m] in [-2.5, 2.5]
        - front_dist: min(NE, NW) [m] in [0.0, 4.0]
        - rear_side_dist: average of rear side sensors (WS, ES) [m] in [0.0, 4.0]

    Outputs:
        - forward_speed: forward drive command in [-1.0, 6.0]
        - steer: differential steering command in [-3.5, 3.5]
                 (positive -> steer left, negative -> steer right)
    """

    def __init__(self, max_dist: float = 4.0):
        self.max_dist = max_dist
        self._build_speed_fis()
        self._build_steer_fis()

    def _build_speed_fis(self):
        self.front_dist = ctrl.Antecedent(np.arange(0.0, self.max_dist + 0.05, 0.05), "front_dist")
        self.rear_side_dist = ctrl.Antecedent(np.arange(0.0, self.max_dist + 0.05, 0.05), "rear_side_dist")
        self.forward_speed = ctrl.Consequent(np.arange(-1.0, 6.05, 0.1), "forward_speed")

        # Front clearance [m]
        self.front_dist["blocked"] = fuzz.trapmf(self.front_dist.universe, [0.0, 0.0, 0.5, 1.0])
        self.front_dist["clear"] = fuzz.trapmf(self.front_dist.universe, [0.7, 1.3, self.max_dist, self.max_dist])

        # Rear side sensor distance [m]: drops below ~1.2m once the rear of the tank is inside the bay
        self.rear_side_dist["inside"] = fuzz.trapmf(self.rear_side_dist.universe, [0.0, 0.0, 0.9, 1.35])
        self.rear_side_dist["entering"] = fuzz.trimf(self.rear_side_dist.universe, [1.0, 1.6, 2.4])
        self.rear_side_dist["outside"] = fuzz.trapmf(self.rear_side_dist.universe, [1.9, 2.6, self.max_dist, self.max_dist])

        # Forward speed output (stop centered at 0.0)
        self.forward_speed["stop"] = fuzz.trimf(self.forward_speed.universe, [-1.0, 0.0, 1.0])
        self.forward_speed["slow"] = fuzz.trimf(self.forward_speed.universe, [0.8, 2.2, 3.6])
        self.forward_speed["medium"] = fuzz.trapmf(self.forward_speed.universe, [3.0, 4.2, 6.0, 6.0])

        f, r, v = self.front_dist, self.rear_side_dist, self.forward_speed
        self.speed_rules = [
            ctrl.Rule(f["blocked"] & r["inside"], v["stop"]),
            ctrl.Rule(f["blocked"] & r["entering"], v["stop"]),
            ctrl.Rule(f["blocked"] & r["outside"], v["stop"]),
            ctrl.Rule(f["clear"] & r["inside"], v["stop"]),
            ctrl.Rule(f["clear"] & r["entering"], v["slow"]),
            ctrl.Rule(f["clear"] & r["outside"], v["medium"]),
        ]
        self.speed_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(self.speed_rules))

    def _build_steer_fis(self):
        self.side_diff = ctrl.Antecedent(np.arange(-2.5, 2.55, 0.05), "side_diff")
        self.steer = ctrl.Consequent(np.arange(-3.5, 3.55, 0.05), "steer")

        # side_diff = left_dist - right_dist
        # Negative -> closer to left car (steer right, negative)
        # Positive -> closer to right car (steer left, positive)
        self.side_diff["much_closer_left"] = fuzz.trapmf(self.side_diff.universe, [-2.5, -2.5, -0.8, -0.35])
        self.side_diff["closer_left"] = fuzz.trimf(self.side_diff.universe, [-0.6, -0.25, -0.05])
        self.side_diff["centered"] = fuzz.trimf(self.side_diff.universe, [-0.15, 0.0, 0.15])
        self.side_diff["closer_right"] = fuzz.trimf(self.side_diff.universe, [0.05, 0.25, 0.6])
        self.side_diff["much_closer_right"] = fuzz.trapmf(self.side_diff.universe, [0.35, 0.8, 2.5, 2.5])

        self.steer["hard_right"] = fuzz.trapmf(self.steer.universe, [-3.5, -3.5, -2.2, -1.2])
        self.steer["right"] = fuzz.trimf(self.steer.universe, [-1.8, -0.9, -0.1])
        self.steer["straight"] = fuzz.trimf(self.steer.universe, [-0.4, 0.0, 0.4])
        self.steer["left"] = fuzz.trimf(self.steer.universe, [0.1, 0.9, 1.8])
        self.steer["hard_left"] = fuzz.trapmf(self.steer.universe, [1.2, 2.2, 3.5, 3.5])

        sd, st = self.side_diff, self.steer
        self.steer_rules = [
            ctrl.Rule(sd["much_closer_left"], st["hard_right"]),
            ctrl.Rule(sd["closer_left"], st["right"]),
            ctrl.Rule(sd["centered"], st["straight"]),
            ctrl.Rule(sd["closer_right"], st["left"]),
            ctrl.Rule(sd["much_closer_right"], st["hard_left"]),
        ]
        self.steer_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(self.steer_rules))

    def compute(self, left_dist: float, right_dist: float, front_dist: float, rear_side_dist: float) -> tuple[float, float]:
        """Return (forward_speed, steer) from distance sensor measurements."""
        self.speed_sim.input["front_dist"] = float(np.clip(front_dist, 0.0, self.max_dist))
        self.speed_sim.input["rear_side_dist"] = float(np.clip(rear_side_dist, 0.0, self.max_dist))
        self.speed_sim.compute()
        fwd = float(self.speed_sim.output["forward_speed"])

        diff = float(np.clip(left_dist - right_dist, -2.5, 2.5))
        self.steer_sim.input["side_diff"] = diff
        self.steer_sim.compute()
        steer = float(self.steer_sim.output["steer"])
        return fwd, steer

    def plot(self, save_path: str | None = None, show: bool = True):
        """Plot all membership functions for BayEntryFuzzyController."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(5, 1, figsize=(9, 12))
        specs = [
            (axes[0], self.front_dist, "Front Clearance (front_dist) [m]"),
            (axes[1], self.rear_side_dist, "Rear Side Distance (rear_side_dist) [m]"),
            (axes[2], self.forward_speed, "Forward Speed Command (forward_speed)"),
            (axes[3], self.side_diff, "Side Balance Error (left - right) [m]"),
            (axes[4], self.steer, "Differential Steering Command (steer)"),
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
            print(f"Saved BayEntryFuzzyController plot to: {save_path}")
        if show:
            plt.show()
        plt.close(fig)


class ParallelAlignFuzzyController:
    """
    Mamdani Fuzzy Inference System for centering a tank longitudinally between
    a front car and a rear car inside a parallel parking spot.

    Input:
        - fb_diff: front_dist - rear_dist [m] in [-2.5, 2.5]
    Output:
        - move_cmd: longitudinal drive command in [-4.0, 4.0]
                    (positive -> forward, negative -> backward)
    """

    def __init__(self):
        self.fb_diff = ctrl.Antecedent(np.arange(-2.5, 2.55, 0.05), "fb_diff")
        self.move_cmd = ctrl.Consequent(np.arange(-4.0, 4.05, 0.05), "move_cmd")

        self.fb_diff["rear_much_more"] = fuzz.trapmf(self.fb_diff.universe, [-2.5, -2.5, -0.7, -0.3])
        self.fb_diff["rear_more"] = fuzz.trimf(self.fb_diff.universe, [-0.5, -0.22, -0.04])
        self.fb_diff["balanced"] = fuzz.trimf(self.fb_diff.universe, [-0.12, 0.0, 0.12])
        self.fb_diff["front_more"] = fuzz.trimf(self.fb_diff.universe, [0.04, 0.22, 0.5])
        self.fb_diff["front_much_more"] = fuzz.trapmf(self.fb_diff.universe, [0.3, 0.7, 2.5, 2.5])

        self.move_cmd["backward"] = fuzz.trapmf(self.move_cmd.universe, [-4.0, -4.0, -2.5, -1.5])
        self.move_cmd["slow_backward"] = fuzz.trimf(self.move_cmd.universe, [-2.0, -1.0, -0.1])
        self.move_cmd["stop"] = fuzz.trimf(self.move_cmd.universe, [-0.4, 0.0, 0.4])
        self.move_cmd["slow_forward"] = fuzz.trimf(self.move_cmd.universe, [0.1, 1.0, 2.0])
        self.move_cmd["forward"] = fuzz.trapmf(self.move_cmd.universe, [1.5, 2.5, 4.0, 4.0])

        d, m = self.fb_diff, self.move_cmd
        self.rules = [
            ctrl.Rule(d["rear_much_more"], m["backward"]),
            ctrl.Rule(d["rear_more"], m["slow_backward"]),
            ctrl.Rule(d["balanced"], m["stop"]),
            ctrl.Rule(d["front_more"], m["slow_forward"]),
            ctrl.Rule(d["front_much_more"], m["forward"]),
        ]
        self.sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(self.rules))

    def compute(self, front_dist: float, rear_dist: float) -> float:
        """Return longitudinal drive command from front and rear distances [m]."""
        diff = float(np.clip(front_dist - rear_dist, -2.5, 2.5))
        self.sim.input["fb_diff"] = diff
        self.sim.compute()
        return float(self.sim.output["move_cmd"])

    def plot(self, save_path: str | None = None, show: bool = True):
        """Plot membership functions for ParallelAlignFuzzyController."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(9, 5.5))
        specs = [
            (axes[0], self.fb_diff, "Front - Rear Distance Error (fb_diff) [m]"),
            (axes[1], self.move_cmd, "Longitudinal Drive Command (move_cmd)"),
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
            print(f"Saved ParallelAlignFuzzyController plot to: {save_path}")
        if show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    ex2_dir = os.path.dirname(os.path.abspath(__file__))
    bay_ctrl = BayEntryFuzzyController()
    bay_ctrl.plot(save_path=os.path.join(ex2_dir, "bay_entry_mf.png"), show=True)
    align_ctrl = ParallelAlignFuzzyController()
    align_ctrl.plot(save_path=os.path.join(ex2_dir, "parallel_align_mf.png"), show=True)
