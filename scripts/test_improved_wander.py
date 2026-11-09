#!/usr/bin/env python3
import math

speed = 0.35
turn_speed = 0.25
turn_omega = math.pi / 4.0  # 45 deg/s = 0.7854 rad/s
turn_duration_steps = int(2.0 / 0.05)  # exactly 40 steps for 90 deg turn

x = 1.2
y = 0.8
yaw = math.pi / 2.0  # 90 deg, North
state = "START"
turn_step = 0

print("Simulating improved target wander logic:")
for step in range(300):
    t = step * 0.05
    cmd_v = 0.0
    cmd_w = 0.0

    if state == "START":
        cmd_v = speed
        cmd_w = 0.0
        if y >= 3.0:
            state = "TURNING"
            turn_step = 0
            print(f"T={t:.2f}s: Started TURNING at x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.1f}°")
    elif state == "TURNING":
        cmd_v = turn_speed
        cmd_w = turn_omega  # Turn left
        turn_step += 1
        if turn_step >= turn_duration_steps:
            state = "WEST"
            yaw = math.pi  # Enforce exact West heading
            cmd_w = 0.0
            print(f"T={t:.2f}s: Finished TURNING at x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.1f}°")
    else:  # WEST
        cmd_v = speed
        cmd_w = 0.0
        if x <= -3.5:
            cmd_v = 0.0

    # Integrate ground-truth pose
    yaw = (yaw + cmd_w * 0.05 + math.pi) % (2.0 * math.pi) - math.pi
    x += cmd_v * math.cos(yaw) * 0.05
    y += cmd_v * math.sin(yaw) * 0.05

    if step % 20 == 0:
        print(f"Step {step:03d} (T={t:04.1f}s): State={state:<7} | Pos=({x:+.2f}, {y:+.2f}) | Yaw={math.degrees(yaw):+6.1f}° | v={cmd_v:.2f}")
