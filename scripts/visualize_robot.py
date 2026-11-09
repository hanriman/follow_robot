#!/usr/bin/env python3
"""
AckBot Physical Layout & Coordinate Visualizer
Reads parameters from rl_ws/configs/robot_params.yaml,
prints a structural schematic, and renders a 3D plot saved to outputs/reports/robot_geometry_3d.png.
"""

from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

CONFIG_PATH = Path(__file__).resolve().parent.parent / "rl_ws" / "configs" / "robot_params.yaml"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "reports"

def load_params():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def print_ascii_diagram(params):
    c = params["chassis"]
    d = params["decks"]
    w = params["wheels"]
    cam = params["camera_rgbd"]
    lid = params["lidar_2d"]

    print("=" * 68)
    print("           ACKBOT (4-WHEEL ACKERMANN ROBOT) STRUCTURAL SPEC          ")
    print("=" * 68)
    print(f"Chassis (L x W x H): {c['length_x']*100:.0f}cm x {c['width_y']*100:.0f}cm | Total Mass: {c['total_mass']} kg")
    print(f"Wheelbase: {w['wheelbase']*100:.0f}cm | Track Width: {w['track_width_front']*100:.0f}cm | Overall Width: {w['overall_width']*100:.0f}cm")
    print(f"Ground Clearance: {c['ground_clearance']*100:.0f}cm | Wheel Diameter: {w['diameter']*100:.0f}cm")
    kin = params.get("kinematics", {})
    if kin:
        print(f"Max Speed: {kin['max_linear_speed']:.2f} m/s | Min Turn Radius: {kin['min_turning_radius']:.3f} m | Max Steer: {np.degrees(kin['max_steering_angle']):.1f}°")
    print("-" * 68)
    print("SIDE VIEW ELEVATION (Z-Axis):")
    print(f"  Z = {lid['pos_z']*100:.0f} cm  [2D LiDAR] (X={lid['pos_x']*100:.0f}cm) 360° Laser Ring")
    print(f"                 || (+4cm mast)")
    print(f"  Z = {d['level_3_z']*100:.0f} cm  ======================================== [Deck 3: Top Plate]")
    print(f"  Z = {cam['pos_z']*100:.0f} cm  |                                      | [Camera] (X={cam['pos_x']*100:.0f}cm)")
    print(f"  Z = {d['level_2_z']*100:.0f} cm  ======================================== [Deck 2: Compute/SBC]")
    print(f"  Z = {d['level_1_z']*100:.0f} cm  ======================================== [Deck 1: Motor/Battery]")
    print(f"  Z = {w['radius']*100:.0f} cm  ( O ) [Rear Axle: X={-w['wheelbase']/2*100:.0f}cm]        ( O ) [Front Axle: X={+w['wheelbase']/2*100:.0f}cm]")
    print(f"  Z = 0 cm   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Ground")
    print("=" * 68)

def plot_deck(ax, x_len, y_len, z, color="lightblue", alpha=0.4, label=None):
    x_half = x_len / 2.0
    y_half = y_len / 2.0
    verts = [
        [-x_half, -y_half, z],
        [ x_half, -y_half, z],
        [ x_half,  y_half, z],
        [-x_half,  y_half, z],
    ]
    poly = Poly3DCollection([verts], color=color, alpha=alpha, edgecolor="navy", linewidth=1.2)
    ax.add_collection3d(poly)
    if label:
        ax.text(0, 0, z + 0.005, label, color="black", fontsize=8, ha="center")

def plot_wheel(ax, x, y, z_center, radius, width, color="dimgray"):
    # Cylinder along Y-axis
    y_half = width / 2.0
    theta = np.linspace(0, 2 * np.pi, 24)
    x_circ = x + radius * np.cos(theta)
    z_circ = z_center + radius * np.sin(theta)

    ax.plot(x_circ, [y - y_half] * len(theta), z_circ, color=color, linewidth=1.5)
    ax.plot(x_circ, [y + y_half] * len(theta), z_circ, color=color, linewidth=1.5)

    # Rim lines
    for i in range(0, len(theta), 4):
        ax.plot([x_circ[i], x_circ[i]], [y - y_half, y + y_half], [z_circ[i], z_circ[i]], color=color, alpha=0.5)

def render_3d(params, save_path):
    c = params["chassis"]
    d = params["decks"]
    w = params["wheels"]
    cam = params["camera_rgbd"]
    lid = params["lidar_2d"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 1. Ground Plane
    g_x, g_y = np.meshgrid(np.linspace(-0.25, 0.25, 2), np.linspace(-0.25, 0.25, 2))
    ax.plot_surface(g_x, g_y, np.zeros_like(g_x), color="lightgray", alpha=0.2)

    # 2. Three Deck Plates
    plot_deck(ax, c["length_x"], c["width_y"], d["level_1_z"], color="royalblue", alpha=0.3, label="Level 1 (6cm)")
    plot_deck(ax, c["length_x"], c["width_y"], d["level_2_z"], color="skyblue", alpha=0.3, label="Level 2 (16cm)")
    plot_deck(ax, c["length_x"], c["width_y"], d["level_3_z"], color="deepskyblue", alpha=0.3, label="Level 3 (26cm)")

    # 3. Corner Standoffs
    for sx in [-c["length_x"] / 2.0 + 0.015, c["length_x"] / 2.0 - 0.015]:
        for sy in [-c["width_y"] / 2.0 + 0.015, c["width_y"] / 2.0 - 0.015]:
            ax.plot([sx, sx], [sy, sy], [d["level_1_z"], d["level_3_z"]], color="darkslategray", linewidth=2.5)

    # 4. Four Wheels
    x_front = w["wheelbase"] / 2.0
    x_rear = -w["wheelbase"] / 2.0
    y_left = w["track_width_front"] / 2.0
    y_right = -w["track_width_front"] / 2.0
    z_axle = w["radius"]

    plot_wheel(ax, x_front, y_left, z_axle, w["radius"], w["width"])
    plot_wheel(ax, x_front, y_right, z_axle, w["radius"], w["width"])
    plot_wheel(ax, x_rear, y_left, z_axle, w["radius"], w["width"])
    plot_wheel(ax, x_rear, y_right, z_axle, w["radius"], w["width"])

    # Axle connecting bars
    ax.plot([x_front, x_front], [y_right, y_left], [z_axle, z_axle], color="black", linestyle="--", alpha=0.6)
    ax.plot([x_rear, x_rear], [y_right, y_left], [z_axle, z_axle], color="black", linestyle="--", alpha=0.6)

    # 5. Front RGB-D Camera
    ax.scatter([cam["pos_x"]], [cam["pos_y"]], [cam["pos_z"]], color="red", s=100, marker="s", label="RGB-D Camera (Z=25cm)")
    ax.quiver(cam["pos_x"], cam["pos_y"], cam["pos_z"], 0.08, 0, 0, color="red", linewidth=2, arrow_length_ratio=0.3)

    # 6. Rear 2D LiDAR (Elevated)
    ax.scatter([lid["pos_x"]], [lid["pos_y"]], [lid["pos_z"]], color="forestgreen", s=120, marker="o", label="2D LiDAR (Z=30cm)")
    # Mast from Level 3 to LiDAR
    ax.plot([lid["pos_x"], lid["pos_x"]], [lid["pos_y"], lid["pos_y"]], [d["level_3_z"], lid["pos_z"]], color="darkgreen", linewidth=3)
    # 360 FOV scan circle (miniature indicator radius 15cm)
    scan_theta = np.linspace(0, 2 * np.pi, 40)
    ax.plot(lid["pos_x"] + 0.15 * np.cos(scan_theta), lid["pos_y"] + 0.15 * np.sin(scan_theta), [lid["pos_z"]] * len(scan_theta),
            color="limegreen", linestyle=":", alpha=0.7, label="LiDAR 360° Scan Plane")

    # Limits and labels
    ax.set_xlim([-0.25, 0.25])
    ax.set_ylim([-0.25, 0.25])
    ax.set_zlim([0.0, 0.35])
    ax.set_xlabel("X (Forward) [m]")
    ax.set_ylabel("Y (Left) [m]")
    ax.set_zlabel("Z (Up) [m]")
    ax.set_title("AckBot 3D Geometric Model (Conforming to spec.txt)", fontsize=12, pad=12)
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc="upper right", fontsize=8)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"\n[+] 3D Geometry plot rendered and saved to: {save_path}")

def main():
    params = load_params()
    print_ascii_diagram(params)
    out_file = OUTPUT_DIR / "robot_geometry_3d.png"
    render_3d(params, out_file)

if __name__ == "__main__":
    main()
