import os
import sys
import time
import numpy as np

# Add ex1 directory (for fuzzy_controller) and fis directory (for vrep, tank, and remoteApi.dll)
EX1_DIR = os.path.dirname(os.path.abspath(__file__))
FIS_DIR = os.path.dirname(EX1_DIR)
os.chdir(FIS_DIR)
for path in (EX1_DIR, FIS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import vrep
from tank import Tank
from fuzzy_controller import BrakingFuzzyController

SENSOR_NAME = "Proximity_sensor"
STOP_CMD_THRESHOLD = 0.4


def main():
    controller = BrakingFuzzyController()

    vrep.simxFinish(-1)
    clientID = vrep.simxStart("127.0.0.1", 19999, True, True, 5000, 5)
    if clientID == -1:
        sys.exit("Could not connect to CoppeliaSim (make sure the simulation is running).")

    print("Connected to CoppeliaSim! Starting fuzzy controller (Exercise 1)...\n")

    tank = None
    try:
        tank = Tank(clientID)

        err, sensor_handle = vrep.simxGetObjectHandle(
            clientID, SENSOR_NAME, vrep.simx_opmode_blocking
        )
        if err != 0:
            sys.exit(f"Proximity sensor '{SENSOR_NAME}' not found (err={err}).")

        # Initialize streaming mode for proximity sensor and tank velocity
        vrep.simxReadProximitySensor(clientID, sensor_handle, vrep.simx_opmode_streaming)
        vel_handle = tank.left_back_handle
        vrep.simxGetObjectVelocity(clientID, vel_handle, vrep.simx_opmode_streaming)
        time.sleep(0.2)

        t0 = time.time()
        stopped_ticks = 0

        while time.time() - t0 < 30.0:
            dt = time.time() - t0

            err_v, lin_vel, _ = vrep.simxGetObjectVelocity(
                clientID, vel_handle, vrep.simx_opmode_buffer
            )
            speed = float(np.linalg.norm(lin_vel[:2])) if err_v == 0 else 0.0

            err_s, detected, point, _, _ = vrep.simxReadProximitySensor(
                clientID, sensor_handle, vrep.simx_opmode_buffer
            )
            dist = float(np.linalg.norm(point)) if (err_s == 0 and detected) else controller.max_dist

            cmd = controller.compute(distance=dist, velocity=speed)

            if cmd <= STOP_CMD_THRESHOLD:
                tank.stop()
                action = "STOP"
                if speed < 0.04 and dist < 1.5:
                    stopped_ticks += 1
                else:
                    stopped_ticks = 0
            else:
                tank.forward(cmd)
                action = f"FWD({cmd:4.2f})"
                stopped_ticks = 0

            print(f"t={dt:5.2f}s | dist={dist:5.2f}m | v={speed:4.2f} m/s | fis_cmd={cmd:5.2f} -> {action}")

            if stopped_ticks >= 5:
                print(f"\nSuccess! Tank stopped safely {dist:.2f}m before the wall.")
                break

            time.sleep(0.1)

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
