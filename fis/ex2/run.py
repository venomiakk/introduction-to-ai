import argparse
import os
import sys
import time
import numpy as np

# Add ex2 directory (for fuzzy_controller) and fis directory (for vrep, tank, and remoteApi.dll)
EX2_DIR = os.path.dirname(os.path.abspath(__file__))
FIS_DIR = os.path.dirname(EX2_DIR)
os.chdir(FIS_DIR)
for path in (EX2_DIR, FIS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import vrep
from tank import Tank
from fuzzy_controller import BayEntryFuzzyController, ParallelAlignFuzzyController

SENSOR_NAMES = ["NE", "NW", "SE", "SW", "EN", "ES", "WN", "WS"]
MAX_DIST = 4.0


def init_sensors(clientID) -> dict[str, int]:
    """Resolve handles for all 8 proximity sensors and start streaming mode."""
    handles = {}
    for name in SENSOR_NAMES:
        err, handle = vrep.simxGetObjectHandle(
            clientID, f"Proximity_sensor_{name}", vrep.simx_opmode_blocking
        )
        if err != 0:
            sys.exit(f"Proximity sensor 'Proximity_sensor_{name}' not found (err={err}).")
        vrep.simxReadProximitySensor(clientID, handle, vrep.simx_opmode_streaming)
        handles[name] = handle
    time.sleep(0.25)
    return handles


def read_sensors(clientID, handles: dict[str, int]) -> dict[str, float]:
    """Read all 8 proximity sensors; return MAX_DIST when nothing is detected."""
    readings = {}
    for name, handle in handles.items():
        err, detected, point, _, _ = vrep.simxReadProximitySensor(
            clientID, handle, vrep.simx_opmode_buffer
        )
        if err == 0 and detected:
            readings[name] = float(np.linalg.norm(point))
        else:
            readings[name] = MAX_DIST
    return readings


def drive_differential(tank: Tank, forward: float, steer: float):
    """Apply differential drive: positive steer turns left, negative steer turns right."""
    tank.go()
    tank.leftvelocity = forward - steer
    tank.rightvelocity = forward + steer
    tank.setVelocity()


def format_sensors(s: dict[str, float]) -> str:
    return " ".join(f"{k}={s[k]:4.2f}" if s[k] < MAX_DIST else f"{k}= ---" for k in SENSOR_NAMES)


def park_left_perpendicular(clientID, tank: Tank, handles: dict[str, int]):
    """Park in the left (West) perpendicular spot using distance sensors and BayEntryFuzzyController."""
    bay_ctrl = BayEntryFuzzyController(max_dist=MAX_DIST)

    # Phase 1: Drive forward until aligned with the left perpendicular gap
    # (NW clears the lower-left cars and SW detects the lower-left car behind-left)
    print("[Phase 1] Searching for the left perpendicular parking gap...")
    tank.forward(5)
    t0 = time.time()
    while time.time() - t0 < 20.0:
        s = read_sensors(clientID, handles)
        print(f"  search | {format_sensors(s)}")
        if s["NW"] >= MAX_DIST and s["SW"] < 3.2:
            print("  -> Left gap aligned! Stopping before turning.")
            tank.stop()
            time.sleep(0.4)
            break
        time.sleep(0.1)

    # Phase 2: Turn left ~90 deg until facing straight into the left bay
    # (front sensors NE & NW both clear after sweeping past the upper-left car)
    print("\n[Phase 2] Turning left into the perpendicular bay...")
    tank.turn_left(3)
    saw_front_car = False
    t0 = time.time()
    while time.time() - t0 < 15.0:
        s = read_sensors(clientID, handles)
        print(f"  turn_L | {format_sensors(s)}")
        if s["NE"] < 2.5 or s["NW"] < 2.5:
            saw_front_car = True
        if saw_front_car and (time.time() - t0) > 1.2 and s["NE"] > 2.6 and s["NW"] > 2.6:
            print("  -> Facing into the left bay!")
            tank.stop()
            time.sleep(0.4)
            break
        time.sleep(0.1)

    # Phase 3: Fuzzy bay entry and centering between flanking cars
    print("\n[Phase 3] Entering left bay with BayEntryFuzzyController...")
    stopped_ticks = 0
    t0 = time.time()
    while time.time() - t0 < 25.0:
        s = read_sensors(clientID, handles)
        left_dist = min(s["WN"], s["WS"])
        right_dist = min(s["EN"], s["ES"])
        front_dist = min(s["NE"], s["NW"])
        rear_side_dist = 0.5 * (s["WS"] + s["ES"])

        fwd, steer = bay_ctrl.compute(left_dist, right_dist, front_dist, rear_side_dist)

        if fwd <= 0.4:
            tank.stop()
            action = "STOP"
            stopped_ticks += 1
        else:
            drive_differential(tank, fwd, steer)
            action = f"FWD={fwd:4.2f} STEER={steer:+5.2f}"
            stopped_ticks = 0

        print(f"  entry  | L={left_dist:4.2f} R={right_dist:4.2f} F={front_dist:4.2f} RS={rear_side_dist:4.2f} -> {action}")

        if stopped_ticks >= 5:
            print("\nSuccess! Tank parked in the left perpendicular spot.")
            break
        time.sleep(0.1)


def park_right_parallel(clientID, tank: Tank, handles: dict[str, int]):
    """Park in the right (East) parallel spot using distance sensors and both fuzzy controllers."""
    bay_ctrl = BayEntryFuzzyController(max_dist=MAX_DIST)
    align_ctrl = ParallelAlignFuzzyController()

    # Phase 1: Drive forward until aligned with the right parallel gap
    # (EN opens over the gap, then ES opens as the tank reaches the center of the gap)
    print("[Phase 1] Searching for the right parallel parking gap...")
    tank.forward(5)
    saw_en_open = False
    t0 = time.time()
    while time.time() - t0 < 20.0:
        s = read_sensors(clientID, handles)
        print(f"  search | {format_sensors(s)}")
        if s["EN"] >= MAX_DIST:
            saw_en_open = True
        if saw_en_open and s["ES"] >= 2.2:
            print("  -> Right gap aligned! Stopping before turning.")
            tank.stop()
            time.sleep(0.4)
            break
        time.sleep(0.1)

    # Phase 2: Turn right ~90 deg until facing straight East into the right gap
    print("\n[Phase 2] Turning right toward the parallel spot...")
    tank.turn_right(3)
    t0 = time.time()
    while time.time() - t0 < 15.0:
        s = read_sensors(clientID, handles)
        print(f"  turn_R | {format_sensors(s)}")
        # Facing East into the gap: front (NE, NW) clear, upper car on left (WN), lower car on right (EN)
        if (time.time() - t0) > 1.2 and s["NE"] > 2.6 and s["NW"] > 2.6 and s["WN"] < 2.2 and s["EN"] < 2.5:
            print("  -> Facing straight into the right gap!")
            tank.stop()
            time.sleep(0.4)
            break
        time.sleep(0.1)

    # Phase 3: Drive into the right gap using BayEntryFuzzyController
    print("\n[Phase 3] Entering right gap with BayEntryFuzzyController...")
    stopped_ticks = 0
    t0 = time.time()
    while time.time() - t0 < 20.0:
        s = read_sensors(clientID, handles)
        left_dist = min(s["WN"], s["WS"])
        right_dist = min(s["EN"], s["ES"])
        front_dist = min(s["NE"], s["NW"])
        rear_side_dist = 0.5 * (s["WS"] + s["ES"])

        fwd, steer = bay_ctrl.compute(left_dist, right_dist, front_dist, rear_side_dist)

        if fwd <= 0.4:
            tank.stop()
            action = "STOP"
            stopped_ticks += 1
        else:
            drive_differential(tank, fwd, steer)
            action = f"FWD={fwd:4.2f} STEER={steer:+5.2f}"
            stopped_ticks = 0

        print(f"  entry  | L={left_dist:4.2f} R={right_dist:4.2f} F={front_dist:4.2f} RS={rear_side_dist:4.2f} -> {action}")

        if stopped_ticks >= 4:
            print("  -> Inside the right gap! Ready to align parallel.")
            break
        time.sleep(0.1)

    # Phase 4: Turn left ~90 deg in place to face North (parallel between front and rear cars)
    print("\n[Phase 4] Turning left to align parallel between front and rear cars...")
    tank.turn_left(3)
    t0 = time.time()
    while time.time() - t0 < 15.0:
        s = read_sensors(clientID, handles)
        print(f"  turn_L | {format_sensors(s)}")
        front_min = min(s["NE"], s["NW"])
        rear_min = min(s["SE"], s["SW"])
        if (time.time() - t0) > 1.2 and front_min < 2.2 and rear_min < 2.2:
            print("  -> Aligned parallel between the two cars!")
            tank.stop()
            time.sleep(0.4)
            break
        time.sleep(0.1)

    # Phase 5: Center longitudinally between front and rear cars with ParallelAlignFuzzyController
    print("\n[Phase 5] Centering between front and rear cars with ParallelAlignFuzzyController...")
    stopped_ticks = 0
    t0 = time.time()
    while time.time() - t0 < 15.0:
        s = read_sensors(clientID, handles)
        front_dist = min(s["NE"], s["NW"])
        rear_dist = min(s["SE"], s["SW"])
        cmd = align_ctrl.compute(front_dist, rear_dist)

        if abs(cmd) <= 0.3:
            tank.stop()
            action = "STOP"
            stopped_ticks += 1
        elif cmd > 0:
            tank.forward(cmd)
            action = f"FWD({cmd:4.2f})"
            stopped_ticks = 0
        else:
            tank.backward(abs(cmd))
            action = f"BWD({abs(cmd):4.2f})"
            stopped_ticks = 0

        print(f"  align  | F={front_dist:4.2f} R={rear_dist:4.2f} -> {action}")

        if stopped_ticks >= 5:
            print("\nSuccess! Tank parked in the right parallel spot.")
            break
        time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description="Exercise 2: Fuzzy parking controller for CoppeliaSim tank.")
    parser.add_argument(
        "--spot",
        choices=["left", "right"],
        default="left",
        help="Which marked parking spot to target: 'left' (perpendicular) or 'right' (parallel).",
    )
    args = parser.parse_args()

    vrep.simxFinish(-1)
    clientID = vrep.simxStart("127.0.0.1", 19999, True, True, 5000, 5)
    if clientID == -1:
        sys.exit("Could not connect to CoppeliaSim (make sure parking_tank.ttt is open and playing).")

    print(f"Connected to CoppeliaSim! Target spot: {args.spot}\n")

    tank = None
    try:
        tank = Tank(clientID)
        handles = init_sensors(clientID)

        if args.spot == "left":
            park_left_perpendicular(clientID, tank, handles)
        else:
            park_right_parallel(clientID, tank, handles)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if tank is not None:
            tank.stop()
            time.sleep(0.1)
        vrep.simxFinish(clientID)
        print("Disconnected.")


if __name__ == "__main__":
    main()
