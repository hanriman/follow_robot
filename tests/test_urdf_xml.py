import xml.etree.ElementTree as ET
from pathlib import Path

ROS_WS_DIR = Path(__file__).resolve().parent.parent / "ros_ws" / "src"


def test_xacro_xml_well_formed():
    """Verify all Xacro XML files in ackbot_description are well-formed XML."""
    desc_dir = ROS_WS_DIR / "ackbot_description" / "urdf"
    xacro_files = list(desc_dir.glob("**/*.xacro"))

    assert len(xacro_files) >= 5, f"Expected at least 5 xacro files, found {len(xacro_files)}"

    for xacro_file in xacro_files:
        tree = ET.parse(xacro_file)
        root = tree.getroot()
        assert root is not None, f"Failed to parse {xacro_file.name}"


def test_sdf_world_well_formed():
    """Verify all Gazebo SDF world files are well-formed XML and contain expected models."""
    worlds_dir = ROS_WS_DIR / "ackbot_gazebo" / "worlds"
    expected_worlds = {
        "follow_arena.sdf": "follow_arena",
        "follow_curriculum.sdf": "follow_curriculum",
        "follow_uturn.sdf": "follow_uturn",
        "follow_clutter.sdf": "follow_clutter",
    }

    for filename, expected_world_name in expected_worlds.items():
        sdf_file = worlds_dir / filename
        assert sdf_file.exists(), f"{sdf_file} not found"

        tree = ET.parse(sdf_file)
        root = tree.getroot()
        assert root.tag == "sdf"
        world = root.find("world")
        assert world is not None
        assert world.attrib.get("name") == expected_world_name

        # Verify overhead spectator camera is present in all curriculum worlds
        if filename != "follow_arena.sdf":
            cam_model = None
            for m in world.findall("model"):
                if m.attrib.get("name") == "overhead_spectator_camera":
                    cam_model = m
                    break
            assert cam_model is not None, f"overhead_spectator_camera missing from {filename}"


def test_xacro_parameters_match_canonical_spec():
    """Verify physical properties declared in Xacro match spec.txt canonical dimensions."""
    wheels_xacro = ROS_WS_DIR / "ackbot_description" / "urdf" / "wheels.xacro"
    tree = ET.parse(wheels_xacro)
    root = tree.getroot()

    props = {}
    for elem in root.findall("{http://www.ros.org/wiki/xacro}property"):
        name = elem.attrib.get("name")
        val = elem.attrib.get("value")
        props[name] = float(val)

    assert props["wheel_radius"] == 0.06
    assert props["wheelbase"] == 0.30
    assert props["track_width"] == 0.30
    assert abs(props["max_steer"] - 0.553953) < 1e-5

    # Verify chassis deck thickness from spec.txt (4 mm)
    chassis_xacro = ROS_WS_DIR / "ackbot_description" / "urdf" / "chassis.xacro"
    c_tree = ET.parse(chassis_xacro)
    c_props = {
        e.attrib.get("name"): float(e.attrib.get("value"))
        for e in c_tree.getroot().findall("{http://www.ros.org/wiki/xacro}property")
    }
    assert c_props["deck_thickness"] == 0.004

    # Verify camera min depth clip from spec.txt (0.15 m)
    cam_xacro = ROS_WS_DIR / "ackbot_description" / "urdf" / "sensors" / "camera.xacro"
    cam_tree = ET.parse(cam_xacro)
    near_elem = cam_tree.getroot().find(".//clip/near")
    assert near_elem is not None
    assert float(near_elem.text) == 0.15


def test_ros_packages_manifests():
    """Verify package.xml files exist and declare correct package names."""
    packages = {
        "ackbot_description": ROS_WS_DIR / "ackbot_description" / "package.xml",
        "ackbot_gazebo": ROS_WS_DIR / "ackbot_gazebo" / "package.xml",
        "ackbot_policy": ROS_WS_DIR / "ackbot_policy" / "package.xml",
    }

    for expected_name, pkg_file in packages.items():
        assert pkg_file.exists(), f"{pkg_file} does not exist"
        tree = ET.parse(pkg_file)
        root = tree.getroot()
        name_elem = root.find("name")
        assert name_elem is not None
        assert name_elem.text.strip() == expected_name


def test_wm_mpc_node_and_launch_files():
    """Verify ROS 2 World Model MPC node script and launch files are well-formed and loadable."""
    node_file = ROS_WS_DIR / "ackbot_policy" / "ackbot_policy" / "wm_mpc_node.py"
    assert node_file.exists(), f"{node_file} not found"

    # Verify python syntax by compiling
    with open(node_file, "r") as f:
        code = f.read()
    compile(code, str(node_file), "exec")

    # Verify launch files
    mpc_launch = ROS_WS_DIR / "ackbot_policy" / "launch" / "mpc.launch.py"
    assert mpc_launch.exists(), f"{mpc_launch} not found"
    with open(mpc_launch, "r") as f:
        compile(f.read(), str(mpc_launch), "exec")

    bringup_launch = ROS_WS_DIR / "ackbot_gazebo" / "launch" / "bringup.launch.py"
    assert bringup_launch.exists(), f"{bringup_launch} not found"
    with open(bringup_launch, "r") as f:
        compile(f.read(), str(bringup_launch), "exec")
