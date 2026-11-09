import os
from glob import glob
from setuptools import find_packages, setup

package_name = "ackbot_policy"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="AckBot Team",
    maintainer_email="developer@example.com",
    description="ROS 2 Python deployment node for trained AckBot PPO policy",
    license="MIT",
    entry_points={
        "console_scripts": [
            "policy_node = ackbot_policy.policy_node:main",
            "wm_mpc_node = ackbot_policy.wm_mpc_node:main",
            "target_wander_node = ackbot_policy.target_wander_node:main",
        ],
    },
)
