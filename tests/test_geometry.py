import yaml
from pathlib import Path
import pytest

CONFIG_PATH = Path(__file__).resolve().parent.parent / "rl_ws" / "configs" / "robot_params.yaml"

@pytest.fixture
def params():
    assert CONFIG_PATH.exists(), f"Configuration file {CONFIG_PATH} does not exist"
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def test_chassis_and_wheelbase(params):
    chassis = params["chassis"]
    wheels = params["wheels"]
    
    assert chassis["length_x"] == 0.30
    assert chassis["width_y"] == 0.20
    assert wheels["wheelbase"] == 0.30
    assert chassis["overhang_front"] == 0.00
    assert chassis["overhang_rear"] == 0.00

def test_deck_elevations(params):
    decks = params["decks"]
    wheels = params["wheels"]
    chassis = params["chassis"]
    
    # Ground clearance matches axle height and wheel radius
    assert chassis["ground_clearance"] == wheels["radius"]
    assert decks["level_1_z"] == 0.06
    assert round(decks["level_2_z"] - decks["level_1_z"], 2) == 0.10
    assert round(decks["level_3_z"] - decks["level_2_z"], 2) == 0.10

def test_track_width_and_clearance(params):
    wheels = params["wheels"]
    chassis = params["chassis"]
    
    assert wheels["track_width_front"] == 0.30
    assert wheels["track_width_rear"] == 0.30
    assert wheels["width"] == 0.06
    assert wheels["overall_width"] == 0.36
    
    # Clearance between inner tire and chassis body wall
    inner_tire_y = (wheels["track_width_front"] / 2.0) - (wheels["width"] / 2.0)
    chassis_wall_y = chassis["width_y"] / 2.0
    gap = inner_tire_y - chassis_wall_y
    assert round(gap, 3) == wheels["chassis_gap"] == 0.02

def test_sensor_placements(params):
    cam = params["camera_rgbd"]
    lidar = params["lidar_2d"]
    decks = params["decks"]
    
    # Camera on front bumper, horizontal
    assert cam["pos_x"] == 0.15
    assert cam["pos_y"] == 0.00
    assert cam["pos_z"] == 0.25
    assert cam["pitch_rad"] == 0.0
    
    # LiDAR above rear axle, elevated above top deck Level 3
    assert lidar["pos_x"] == -0.15
    assert lidar["pos_y"] == 0.00
    assert lidar["pos_z"] == 0.30
    assert lidar["pos_z"] > decks["level_3_z"]
    assert lidar["fov_deg"] == 360.0
    assert lidar["range_max"] == 3.00

def test_kinematics_and_steering(params):
    import math
    kin = params["kinematics"]
    wheels = params["wheels"]
    
    assert kin["max_linear_speed"] == 0.8
    assert kin["min_turning_radius"] == 0.485
    
    # Ackermann geometry: tan(delta) = L / R
    expected_delta = math.atan(wheels["wheelbase"] / kin["min_turning_radius"])
    assert math.isclose(kin["max_steering_angle"], expected_delta, rel_tol=1e-4)
