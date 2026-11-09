#!/usr/bin/env python3
"""
Test script to log exact poses of human_target and ackbot from Gazebo dynamic_pose/info.
"""

import subprocess
import time
import re
import math


def parse_pose_msg(block):
    m_name = re.search(r'name:\s*"([^"]+)"', block)
    if not m_name:
        return None, None
    name = m_name.group(1)
    if name not in ("human_target", "ackbot"):
        return None, None

    px, py, pz = 0.0, 0.0, 0.0
    m_pos = re.search(r'position\s*\{([^}]*)\}', block)
    if m_pos:
        pos_str = m_pos.group(1)
        mx = re.search(r'x:\s*([+-]?\d+\.?\d*(?:e[+-]?\d+)?)', pos_str)
        my = re.search(r'y:\s*([+-]?\d+\.?\d*(?:e[+-]?\d+)?)', pos_str)
        mz = re.search(r'z:\s*([+-]?\d+\.?\d*(?:e[+-]?\d+)?)', pos_str)
        if mx:
            px = float(mx.group(1))
        if my:
            py = float(my.group(1))
        if mz:
            pz = float(mz.group(1))

    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    m_ori = re.search(r'orientation\s*\{([^}]*)\}', block)
    if m_ori:
        ori_str = m_ori.group(1)
        mx = re.search(r'x:\s*([+-]?\d+\.?\d*(?:e[+-]?\d+)?)', ori_str)
        my = re.search(r'y:\s*([+-]?\d+\.?\d*(?:e[+-]?\d+)?)', ori_str)
        mz = re.search(r'z:\s*([+-]?\d+\.?\d*(?:e[+-]?\d+)?)', ori_str)
        mw = re.search(r'w:\s*([+-]?\d+\.?\d*(?:e[+-]?\d+)?)', ori_str)
        if mx:
            qx = float(mx.group(1))
        if my:
            qy = float(my.group(1))
        if mz:
            qz = float(mz.group(1))
        if mw:
            qw = float(mw.group(1))

    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw_deg = math.degrees(math.atan2(siny, cosy))
    return name, (px, py, pz, yaw_deg)


def main():
    print("Launching accurate multi-model tracker...")
    proc = subprocess.Popen(
        ["gz", "topic", "-e", "-t", "/world/follow_curriculum/dynamic_pose/info"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    t0 = time.time()
    try:
        current_block = ""
        last_print = 0
        poses = {}
        while time.time() - t0 < 20.0:
            line = proc.stdout.readline()
            if not line:
                break
            if re.match(r"^\s*pose\s*\{", line):
                if current_block:
                    name, data = parse_pose_msg(current_block)
                    if name:
                        poses[name] = data
                current_block = line
            else:
                current_block += line

            now = time.time() - t0
            if now - last_print >= 0.5 and "human_target" in poses:
                last_print = now
                ht = poses.get("human_target", (0, 0, 0, 0))
                ab = poses.get("ackbot", (0, 0, 0, 0))
                dist = math.hypot(ht[0] - ab[0], ht[1] - ab[1])
                print(
                    f"T={now:04.1f}s | "
                    f"Human: pos=({ht[0]:+.2f}, {ht[1]:+.2f}), yaw={ht[3]:+5.1f}° | "
                    f"AckBot: pos=({ab[0]:+.2f}, {ab[1]:+.2f}), yaw={ab[3]:+5.1f}° | "
                    f"Dist={dist:.2f}m"
                )
    finally:
        proc.kill()


if __name__ == "__main__":
    main()
