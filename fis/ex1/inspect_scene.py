import os
import sys
import time
import numpy as np

# The parent fis directory holds vrep.py, tank.py, and remoteApi.dll
EX1_DIR = os.path.dirname(os.path.abspath(__file__))
FIS_DIR = os.path.dirname(EX1_DIR)
os.chdir(FIS_DIR)
if FIS_DIR not in sys.path:
    sys.path.insert(0, FIS_DIR)

import vrep
from tank import Tank

SENSOR_NAME = "Proximity_sensor"


def main():
    vrep.simxFinish(-1)
    clientID = vrep.simxStart("127.0.0.1", 19999, True, True, 5000, 5)
    if clientID == -1:
        sys.exit("Could not connect to CoppeliaSim (make sure the simulation is running).")

    print("Connected to CoppeliaSim!")

    tank = None
    try:
        tank = Tank(clientID)

        err, sensor_handle = vrep.simxGetObjectHandle(
            clientID, SENSOR_NAME, vrep.simx_opmode_blocking
        )
        if err != 0:
            sys.exit(f"Proximity sensor '{SENSOR_NAME}' not found (err={err}).")

        vrep.simxReadProximitySensor(clientID, sensor_handle, vrep.simx_opmode_streaming)
        vel_handle = tank.left_back_handle
        vrep.simxGetObjectVelocity(clientID, vel_handle, vrep.simx_opmode_streaming)
        time.sleep(0.2)

        cmd_speed = 10
        stop_dist = 1.5
        max_duration = 20.0
        print(f"\nDriving forward with tank.forward({cmd_speed}) until wall distance < {stop_dist}m (Ctrl+C to abort)...\n")
        tank.forward(cmd_speed)

        t0 = time.time()
        stopped = False
        stop_time = None
        while time.time() - t0 < max_duration:
            dt = time.time() - t0

            err_v, lin_vel, _ = vrep.simxGetObjectVelocity(
                clientID, vel_handle, vrep.simx_opmode_buffer
            )
            speed = float(np.linalg.norm(lin_vel[:2])) if err_v == 0 else float("nan")

            err_s, detected, point, _, _ = vrep.simxReadProximitySensor(
                clientID, sensor_handle, vrep.simx_opmode_buffer
            )
            dist = float(np.linalg.norm(point)) if (err_s == 0 and detected) else None
            dist_str = f"{dist:5.2f}m" if dist is not None else "   ---"

            state = "STOP" if stopped else f"cmd={tank.leftvelocity}"
            print(f"t={dt:5.2f}s | v={speed:4.2f} m/s ({state:6s}) | {SENSOR_NAME}={dist_str}")

            if not stopped and dist is not None and dist < stop_dist:
                print(f"\n>>> Calling tank.stop() at dist={dist:.2f}m, v={speed:.2f} m/s <<<\n")
                tank.stop()
                stopped = True
                stop_time = time.time()

            if stopped and (time.time() - stop_time) > 2.0:
                break

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if tank is not None:
            print("\nStopping tank (tank.stop())...")
            tank.stop()
            time.sleep(0.2)
        vrep.simxFinish(clientID)
        print("Disconnected.")


if __name__ == "__main__":
    main()
