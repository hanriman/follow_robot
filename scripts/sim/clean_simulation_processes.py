#!/usr/bin/env python3
"""
Safely terminate simulation processes inside Docker without self-killing.
"""

import os
import subprocess

my_pid = os.getpid()
parent_pid = os.getppid()

patterns = [
    "bringup.launch.py",
    "sim.launch.py",
    "gz sim",
    "ruby.*gz",
    "wm_mpc_node",
    "policy_node",
    "target_wander_node",
    "parameter_bridge",
    "foxglove_bridge",
    "robot_state_publisher",
    "check_tf",
    "test_human_motion",
    "record_simulation_media",
    "scripts/",
    "Probe",
]

killed = 0
for pattern in patterns:
    res = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    for line in res.stdout.strip().splitlines():
        if line.strip():
            try:
                pid = int(line.strip())
                if pid not in (my_pid, parent_pid, 1):
                    os.kill(pid, 9)
                    killed += 1
            except (ValueError, ProcessLookupError, PermissionError):
                pass

print(f"Cleaned {killed} simulation processes.")
