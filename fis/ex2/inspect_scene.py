import os
import sys
import time
import numpy as np

# Parent fis directory holds vrep.py, tank.py, and remoteApi.dll
EX2_DIR = os.path.dirname(os.path.abspath(__file__))
FIS_DIR = os.path.dirname(EX2_DIR)
os.chdir(FIS_DIR)
if FIS_DIR not in sys.path:
    sys.path.insert(0, FIS_DIR)

import vrep
from tank import Tank

PROXIMITY_SENSORS = ["NE", "NW", "SE", "SW", "EN", "ES", "WN", "WS"]


def main():
    vrep.simxFinish(-1)
    clientID = vrep.simxStart("127.0.0.1", 19999, True, True, 5000, 5)
    if clientID == -1:
        sys.exit("Could not connect to CoppeliaSim (make sure parking_tank.ttt is open and playing).")

    print("Connected to CoppeliaSim!\n")

    tank = None
    try:
        # 1. List scene objects (skip repetitive tank suspension/joint parts)
        err, handles, _, _, names = vrep.simxGetObjectGroupData(
            clientID, vrep.sim_appobj_object_type, 0, vrep.simx_opmode_blocking
        )
        if err == 0:
            skip_prefixes = ("sj_", "suspension", "track", "wheel", "left_", "right_", "Cuboid_wheel")
            filtered = [n for n in names if not n.startswith(skip_prefixes)]
            print("Scene objects (filtered):")
            print(filtered)
            print()

        # 2. Initialize tank and proximity sensors
        tank = Tank(clientID)

        sensor_handles = {}
        for name in PROXIMITY_SENSORS:
            err, handle = vrep.simxGetObjectHandle(
                clientID, f"Proximity_sensor_{name}", vrep.simx_opmode_blocking
            )
            if err == 0:
                sensor_handles[name] = handle
                vrep.simxReadProximitySensor(clientID, handle, vrep.simx_opmode_streaming)
            else:
                print(f"Warning: Proximity_sensor_{name} not found (err={err})")

        # Print positions of the two marked parking spots (for diagnostic understanding only)
        for spot_name in ("5mX5mMetallicFloor", "5mX5mMetallicFloor0"):
            err_h, h = vrep.simxGetObjectHandle(clientID, spot_name, vrep.simx_opmode_blocking)
            if err_h == 0:
                err_p, p = vrep.simxGetObjectPosition(clientID, h, -1, vrep.simx_opmode_blocking)
                if err_p == 0:
                    print(f"Marked spot {spot_name}: ({p[0]:.2f}, {p[1]:.2f})")

        # Print initial tank position and orientation using left_back_handle
        err_p, pos = vrep.simxGetObjectPosition(clientID, tank.left_back_handle, -1, vrep.simx_opmode_blocking)
        err_o, ori = vrep.simxGetObjectOrientation(clientID, tank.left_back_handle, -1, vrep.simx_opmode_blocking)
        if err_p == 0 and err_o == 0:
            print(f"Initial tank ref pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                  f"ori (deg): ({np.degrees(ori[0]):.1f}, {np.degrees(ori[1]):.1f}, {np.degrees(ori[2]):.1f})\n")

        vrep.simxGetObjectPosition(clientID, tank.left_back_handle, -1, vrep.simx_opmode_streaming)
        time.sleep(0.25)

        # 3. Drive forward past both parking gaps and log all 8 proximity sensors
        cmd_speed = 6
        duration = 14.0
        print(f"Driving forward with tank.forward({cmd_speed}) for {duration}s (Ctrl+C to abort)...\n")
        tank.forward(cmd_speed)

        t0 = time.time()
        while time.time() - t0 < duration:
            dt = time.time() - t0
            err_p, pos = vrep.simxGetObjectPosition(clientID, tank.left_back_handle, -1, vrep.simx_opmode_buffer)
            pos_str = f"y={pos[1]:5.2f}" if err_p == 0 else "y=  ---"
            parts = []
            for name, handle in sensor_handles.items():
                err_s, detected, point, _, _ = vrep.simxReadProximitySensor(
                    clientID, handle, vrep.simx_opmode_buffer
                )
                if err_s == 0:
                    if detected:
                        dist = float(np.linalg.norm(point))
                        parts.append(f"{name}={dist:4.2f}")
                    else:
                        parts.append(f"{name}= ---")
            print(f"t={dt:4.1f}s | {pos_str} | " + " ".join(parts))
            time.sleep(0.4)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if tank is not None:
            print("\nStopping tank...")
            tank.stop()
            time.sleep(0.1)
        vrep.simxFinish(clientID)
        print("Disconnected.")


if __name__ == "__main__":
    main()
