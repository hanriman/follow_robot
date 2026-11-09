#!/usr/bin/env python3
"""
Autonomous Motion Node for Human Target Beacon and Distractors in Gazebo Simulation.
Supports Curriculum Scenarios:
1. 'open'    : Wide arena forward wander with boundary repulsion
2. 'corner'  : Guided path around L-corner blind bend (triggers camera occlusion)
3. 'uturn'   : Corridor walk to dead-end, 180-deg pivot, and reverse traversal past robot
4. 'clutter' : Multi-obstacle pillar avoidance with dual-pedestrian Re-ID crossing
"""

import math
import random
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class TargetWanderNode(Node):
    def __init__(self):
        super().__init__("target_wander_node")

        # Declare parameters
        self.declare_parameter("speed", 0.35)
        self.declare_parameter("arena_boundary", 3.8)
        self.declare_parameter("update_rate", 20.0)
        self.declare_parameter("scenario", "open")

        self.speed = float(self.get_parameter("speed").value)
        self.boundary = float(self.get_parameter("arena_boundary").value)
        self.scenario = str(self.get_parameter("scenario").value).lower()
        rate = float(self.get_parameter("update_rate").value)
        self.dt = 1.0 / rate

        # Publishers
        self.pub_cmd = self.create_publisher(Twist, "/human_target/cmd_vel", 10)
        self.pub_distractor = self.create_publisher(Twist, "/distractor_human/cmd_vel", 10)

        # Static obstacle pillars (matching AckBotFollow-v0 Gymnasium parameters)
        self.pillars = [
            (1.8, 1.8, 0.35),
            (-2.0, 1.5, 0.40),
            (2.0, -2.0, 0.45),
            (-2.2, -1.8, 0.35),
            (0.0, 3.0, 0.40),
            (0.0, -3.0, 0.40),
        ]

        # Primary target internal state
        if self.scenario == "corner":
            self.x = 1.2
            self.y = 0.8
            self.yaw = math.pi / 2.0  # Facing North (+Y)
        elif self.scenario == "uturn":
            self.x = 0.5
            self.y = 0.0
            self.yaw = 0.0  # Facing East (+X)
        elif self.scenario == "clutter":
            self.x = 0.0
            self.y = 0.8
            self.yaw = math.pi / 2.0  # Facing North
        else:
            self.x = 2.0
            self.y = 0.0
            self.yaw = 0.0

        self.current_omega = 0.0
        self.target_omega = 0.0
        self.steps_until_turn_change = 0
        self.scenario_substate = "START"
        self.timer_counter = 0

        # Calibrated 90-deg left corner maneuver (Stage 2)
        self.corner_turn_step = 0
        self.corner_turn_steps = int(2.0 / self.dt)  # 40 steps = 2.0s
        self.corner_turn_omega = (math.pi / 2.0) / 2.0  # pi/4 rad/s
        self.corner_turn_speed = 0.25

        # Calibrated 180-deg U-turn maneuver (Stage 3)
        self.uturn_step = 0
        self.uturn_steps = int(2.5 / self.dt)  # 50 steps = 2.5s
        self.uturn_omega = math.pi / 2.5  # pi / 2.5 rad/s = 180 deg in 2.5s
        self.uturn_speed = 0.22  # produces lateral arc of radius r = 0.22 / (pi/2.5) ~ 0.175m -> lane offset ~0.35m

        # Secondary Distractor State (Stage 4 Clutter)
        self.distractor_x = -1.8
        self.distractor_y = 0.0
        self.distractor_yaw = 0.0
        self.distractor_speed = 0.28
        self.distractor_omega = 0.0
        self.distractor_target_omega = 0.0
        self.distractor_steps_change = 0

        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info(
            f"Human Target Active | Scenario: '{self.scenario}' | Speed: {self.speed} m/s | Rate: {rate:.0f} Hz"
        )

    def timer_callback(self):
        twist = Twist()

        # ======================================================================
        # Scenario 2: Corner (L-Corner Blind Bend Occlusion)
        # ======================================================================
        if self.scenario == "corner":
            if self.scenario_substate == "START":
                twist.linear.x = self.speed
                twist.angular.z = 0.0
                if self.y >= 3.0:
                    self.scenario_substate = "TURNING"
                    self.corner_turn_step = 0
            elif self.scenario_substate == "TURNING":
                twist.linear.x = self.corner_turn_speed
                twist.angular.z = self.corner_turn_omega  # Turn left smoothly
                self.corner_turn_step += 1
                if self.corner_turn_step >= self.corner_turn_steps:
                    self.scenario_substate = "WEST"
                    self.yaw = math.pi  # Exact West heading
                    twist.angular.z = 0.0
            else:  # WEST
                twist.linear.x = self.speed
                twist.angular.z = 0.0
                if self.x <= -3.8:
                    twist.linear.x = 0.0

        # ======================================================================
        # Scenario 3: U-Turn (Dead-End Corridor Pivot & Yielding)
        # ======================================================================
        elif self.scenario == "uturn":
            if self.scenario_substate == "START":
                twist.linear.x = self.speed
                twist.angular.z = 0.0
                if self.x >= 2.6:
                    self.scenario_substate = "HALT"
                    self.timer_counter = 20  # 1.0s pause at dead end
            elif self.scenario_substate == "HALT":
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.timer_counter -= 1
                if self.timer_counter <= 0:
                    self.scenario_substate = "PIVOT"
                    self.uturn_step = 0
            elif self.scenario_substate == "PIVOT":
                # Smooth counter-clockwise arc turning 180 deg into the North lane (y ~ +0.40m)
                twist.linear.x = self.uturn_speed
                twist.angular.z = self.uturn_omega
                self.uturn_step += 1
                if self.uturn_step >= self.uturn_steps:
                    self.scenario_substate = "WALK_BACK"
                    self.yaw = math.pi  # Exact West heading
                    twist.angular.z = 0.0
            else:  # WALK_BACK
                twist.linear.x = self.speed
                twist.angular.z = 0.0
                if self.x <= -3.2:
                    twist.linear.x = 0.0  # Reached corridor exit

        # ======================================================================
        # Scenario 4: Clutter & Open (Obstacle Avoidance & Multi-Agent Re-ID)
        # ======================================================================
        else:
            self.steps_until_turn_change -= 1
            if self.steps_until_turn_change <= 0:
                self.target_omega = random.uniform(-0.35, 0.35)
                self.steps_until_turn_change = random.randint(25, 60)

            self.current_omega += (self.target_omega - self.current_omega) * 0.1

            # Boundary repulsion
            turn_override = 0.0
            override_weight = 0.0

            if abs(self.x) > self.boundary or abs(self.y) > self.boundary:
                desired_yaw = math.atan2(-self.y, -self.x)
                diff = (desired_yaw - self.yaw + math.pi) % (2.0 * math.pi) - math.pi
                turn_override = math.copysign(0.8, diff)
                override_weight = 0.8

            if self.scenario == "clutter":
                for ox, oy, r in self.pillars:
                    dx = self.x - ox
                    dy = self.y - oy
                    dist = math.hypot(dx, dy)
                    margin = r + 0.65
                    if dist < margin:
                        away_yaw = math.atan2(dy, dx)
                        diff = (away_yaw - self.yaw + math.pi) % (2.0 * math.pi) - math.pi
                        turn_override = math.copysign(1.0, diff)
                        override_weight = 1.0
                        break

            effective_omega = (1.0 - override_weight) * self.current_omega + override_weight * turn_override
            twist.linear.x = self.speed
            twist.angular.z = effective_omega

            # Publish Distractor Pedestrian in Clutter mode
            if self.scenario == "clutter":
                dist_twist = self._compute_distractor_cmd()
                self.pub_distractor.publish(dist_twist)

        # Integrate primary target estimated position
        self.yaw = (self.yaw + twist.angular.z * self.dt + math.pi) % (2.0 * math.pi) - math.pi
        self.x += twist.linear.x * math.cos(self.yaw) * self.dt
        self.y += twist.linear.x * math.sin(self.yaw) * self.dt
        self.pub_cmd.publish(twist)

    def _compute_distractor_cmd(self) -> Twist:
        """Compute autonomous trajectory for secondary distractor pedestrian in Clutter mode."""
        dtwist = Twist()

        self.distractor_steps_change -= 1
        if self.distractor_steps_change <= 0:
            self.distractor_target_omega = random.uniform(-0.3, 0.3)
            self.distractor_steps_change = random.randint(20, 50)

        self.distractor_omega += (self.distractor_target_omega - self.distractor_omega) * 0.1

        turn_override = 0.0
        override_weight = 0.0

        # Arena boundary repulsion
        if abs(self.distractor_x) > self.boundary or abs(self.distractor_y) > self.boundary:
            desired_yaw = math.atan2(-self.distractor_y, -self.distractor_x)
            diff = (desired_yaw - self.distractor_yaw + math.pi) % (2.0 * math.pi) - math.pi
            turn_override = math.copysign(0.8, diff)
            override_weight = 0.8

        # Pillar repulsion
        for ox, oy, r in self.pillars:
            dx = self.distractor_x - ox
            dy = self.distractor_y - oy
            dist = math.hypot(dx, dy)
            margin = r + 0.65
            if dist < margin:
                away_yaw = math.atan2(dy, dx)
                diff = (away_yaw - self.distractor_yaw + math.pi) % (2.0 * math.pi) - math.pi
                turn_override = math.copysign(1.0, diff)
                override_weight = 1.0
                break

        effective_w = (1.0 - override_weight) * self.distractor_omega + override_weight * turn_override
        dtwist.linear.x = self.distractor_speed
        dtwist.angular.z = effective_w

        # Integrate distractor pose
        self.distractor_yaw = (self.distractor_yaw + dtwist.angular.z * self.dt + math.pi) % (2.0 * math.pi) - math.pi
        self.distractor_x += dtwist.linear.x * math.cos(self.distractor_yaw) * self.dt
        self.distractor_y += dtwist.linear.x * math.sin(self.distractor_yaw) * self.dt
        return dtwist


def main(args=None):
    rclpy.init(args=args)
    node = TargetWanderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
