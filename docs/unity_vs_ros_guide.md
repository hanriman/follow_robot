# Unity ML-Agents to ROS 2 & Gazebo: Visual & Concept Mapping

If you have trained reinforcement learning agents using **Unity (ML-Agents)**, this guide maps every familiar Unity concept directly to the ROS 2 and Gazebo robotics stack.

---

## 1. Direct Concept Mapping

| Unity ML-Agents Concept | ROS 2 & Gazebo Equivalent | Role in This Project |
| :--- | :--- | :--- |
| **Unity Scene View** (3D Viewport) | **Gazebo Sim (Harmonic)** | The 3D physical world with rigid bodies, friction, collision geometry, lighting, and ground terrain. |
| **Unity Game View / Cameras** | **Gazebo Sensor Rendering + RViz 2 / Foxglove** | The actual view from sensors. In ROS, sensor streams are published as topics (`/camera/image_raw`, `/scan`). |
| **GameObject Hierarchy** | **URDF / Xacro Model** | Defines parent-child links, joints, wheels, deck plates, and sensors with real physical inertias. |
| **WheelColliders / Physics** | **Ackermann Steering Plugin / `ros2_control`** | Real-world tire friction, turning radius, and Ackermann steering kinematics. |
| **RayPerceptionSensor** | **2D LiDAR Sensor Plugin (`/scan`)** | Casts 360 laser rays to measure distance to obstacles in meters (max 3m). |
| **Camera Sensor Component** | **RGB-D Camera Plugin (`/camera/depth`)** | Streams 640x480 color and metric depth images. |
| **ML-Agents `Agent.CollectObservations()`** | **Gymnasium `env.observation_space`** | Bundles LiDAR distances, camera depth/features, relative target vector, and vehicle velocity into an observation tensor. |
| **ML-Agents `Agent.OnActionReceived()`** | **Gymnasium `env.step(action)`** | Applies continuous velocity $v$ and steering angle $\delta$ to drive wheels and steering knuckles. |
| **ML-Agents Reward Functions** | **`reward_wrappers.py`** | Rewards target tracking distance, heading alignment, and penalizes collisions. |
| **TensorBoard in ML-Agents** | **`tensorboard --logdir outputs/experiments`** | Tracks episodic returns, policy loss, value loss, and tracking error over time. |

---

## 2. The Visual Tools You Will See

In Unity, everything was inside the Unity Editor window. In robotics, the system is modular:

1. **Live Gymnasium Viewer (`ackbot-eval --render` / `ackbot-view-gym`):**
   * Real-time 2D top-down view with live Ackermann steering, 360° LiDAR rays, and front RGB-D camera viewport right on your desktop.
   * Runs the genuine mathematical simulation and neural network policy.
2. **Gazebo Sim Window (Phase 4):**
   * The heavy physics simulator showing the robot in a realistic 3D arena.
3. **RViz 2 / Foxglove Studio (Phase 4):**
   * The robotics visualization dashboard. Displays the 3D robot model, colored LiDAR laser point clouds, and the front camera stream side-by-side.
4. **TensorBoard (`outputs/experiments`):**
   * Training progress curves (just like Unity ML-Agents training logs).
