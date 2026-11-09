#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

class TfChecker(Node):
    def __init__(self):
        super().__init__("tf_checker")
        self.sub = self.create_subscription(TFMessage, "/tf", self.cb, 10)

    def cb(self, msg: TFMessage):
        for t in msg.transforms:
            if "human" in t.child_frame_id.lower() or "target" in t.child_frame_id.lower():
                p = t.transform.translation
                print(f"TF human target: frame={t.child_frame_id}, pos=({p.x:.2f}, {p.y:.2f}, {p.z:.2f})")
                rclpy.shutdown()

rclpy.init()
node = TfChecker()
rclpy.spin(node)
