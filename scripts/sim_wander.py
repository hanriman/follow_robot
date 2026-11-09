#!/usr/bin/env python3
import math

speed = 0.35
dt = 0.05

x = 1.2
y = 0.8
yaw = math.pi / 2.0
scenario_substate = "START"

for step in range(300):
    t = step * dt
    lin_x = 0.0
    ang_z = 0.0

    if scenario_substate == "START":
        lin_x = speed
        ang_z = 0.0
        if y >= 2.5:
            scenario_substate = "TURNING"
            print(f"T={t:.2f}s: Transitioned to TURNING at x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.1f}°")
    elif scenario_substate == "TURNING":
        lin_x = speed * 0.7
        ang_z = 0.8
        if abs(yaw - math.pi) < 0.15 or yaw < -3.0:
            yaw = math.pi
            scenario_substate = "WEST"
            print(f"T={t:.2f}s: Transitioned to WEST at x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.1f}°")
    else: # WEST
        lin_x = speed
        ang_z = 0.0
        if x <= -3.5:
            lin_x = 0.0

    # Pose integration in target_wander_node.py:
    yaw = (yaw + ang_z * dt + math.pi) % (2.0 * math.pi) - math.pi
    x += lin_x * math.cos(yaw) * dt
    y += lin_x * math.sin(yaw) * dt

    if step % 20 == 0:
        print(f"Step {step:03d} (T={t:04.1f}s): Substate={scenario_substate} | x={x:+.2f}, y={y:+.2f}, yaw={math.degrees(yaw):+6.1f}° | cmd=[v={lin_x:.2f}, w={ang_z:.2f}]")
