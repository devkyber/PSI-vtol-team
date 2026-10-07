"""Write the Gazebo model, the world and the PX4 start-up settings of one simulation case.

The vehicle is the placeholder of vehicle.yaml: one rigid body, four lift rotors, one pusher,
and lift/drag surfaces for the wing, the flat plate between the booms and the side area of
the fuselage. No fin and no control surfaces.

Body axes are Gazebo's (x forward, y left, z up). PX4's are x forward, y right, z down, so a
y or z coordinate changes sign on the way into a PX4 parameter.
"""
import math
from pathlib import Path

G = 9.80665
RHO_REF = 1.2041                # density the motor constants are quoted at
LIFT_MAX_RADPS = 1500.0         # must equal SIM_GZ_EC_MAX1..4 of the airframe
PUSHER_MAX_RADPS = 3500.0       # must equal SIM_GZ_EC_MAX5
OSWALD = 0.85
MODEL_NAME = "psi_qp"

# Sensors of PX4's standard_vtol model, unchanged.
_NOISE = """<noise type="gaussian"><mean>0</mean><stddev>{sd}</stddev>
                <dynamic_bias_stddev>{bias}</dynamic_bias_stddev>
                <dynamic_bias_correlation_time>{corr}</dynamic_bias_correlation_time></noise>"""
_GYRO = _NOISE.format(sd="0.0003394", bias="3.8785e-05", corr="1000")
_ACCEL = _NOISE.format(sd="0.004", bias="0.006", corr="300")
SENSORS = f"""
      <sensor name="imu_sensor" type="imu">
        <gz_frame_id>base_link</gz_frame_id>
        <always_on>1</always_on>
        <update_rate>250</update_rate>
        <imu>
          <angular_velocity><x>{_GYRO}</x><y>{_GYRO}</y><z>{_GYRO}</z></angular_velocity>
          <linear_acceleration><x>{_ACCEL}</x><y>{_ACCEL}</y><z>{_ACCEL}</z></linear_acceleration>
        </imu>
      </sensor>
      <sensor name="air_pressure_sensor" type="air_pressure">
        <gz_frame_id>base_link</gz_frame_id>
        <always_on>1</always_on>
        <update_rate>50</update_rate>
        <air_pressure><pressure><noise type="gaussian"><mean>0</mean><stddev>0.01</stddev></noise></pressure></air_pressure>
      </sensor>
      <sensor name="magnetometer_sensor" type="magnetometer">
        <gz_frame_id>base_link</gz_frame_id>
        <always_on>1</always_on>
        <update_rate>100</update_rate>
        <magnetometer>
          <x><noise type="gaussian"><stddev>0.0001</stddev></noise></x>
          <y><noise type="gaussian"><stddev>0.0001</stddev></noise></y>
          <z><noise type="gaussian"><stddev>0.0001</stddev></noise></z>
        </magnetometer>
      </sensor>
      <sensor name="navsat_sensor" type="navsat">
        <gz_frame_id>base_link</gz_frame_id>
        <always_on>1</always_on>
        <update_rate>30</update_rate>
      </sensor>"""

# Gazebo systems PX4 needs, without the rendering ones (no camera on this model).
SERVER_CONFIG = """<server_config>
  <plugins>
    <plugin entity_name="*" entity_type="world" filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin entity_name="*" entity_type="world" filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin entity_name="*" entity_type="world" filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin entity_name="*" entity_type="world" filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin entity_name="*" entity_type="world" filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin entity_name="*" entity_type="world" filename="gz-sim-air-pressure-system" name="gz::sim::systems::AirPressure"/>
    <plugin entity_name="*" entity_type="world" filename="gz-sim-air-speed-system" name="gz::sim::systems::AirSpeed"/>
    <plugin entity_name="*" entity_type="world" filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>
    <plugin entity_name="*" entity_type="world" filename="gz-sim-magnetometer-system" name="gz::sim::systems::Magnetometer"/>{gusts}
  </plugins>
</server_config>
"""

# Gusts. This system goes in the server configuration, not in the world file: Gazebo drops the
# whole default list above as soon as a world names a system of its own.
GUSTS = """
    <plugin entity_name="*" entity_type="world" filename="gz-sim-wind-effects-system" name="gz::sim::systems::WindEffects">
      <force_approximation_scaling_factor>0</force_approximation_scaling_factor>
      <horizontal>
        <magnitude>
          <time_for_rise>2</time_for_rise>
          <sin><amplitude_percent>{amplitude}</amplitude_percent><period>12</period></sin>
          <noise type="gaussian"><mean>0</mean><stddev>0.05</stddev></noise>
        </magnitude>
        <direction>
          <time_for_rise>2</time_for_rise>
          <sin><amplitude>0.2</amplitude><period>17</period></sin>
          <noise type="gaussian"><mean>0</mean><stddev>0.02</stddev></noise>
        </direction>
      </horizontal>
      <vertical><noise type="gaussian"><mean>0</mean><stddev>0.02</stddev></noise></vertical>
    </plugin>"""


def derived(p):
    """Quantities that follow from the vehicle numbers. Also used to report a case."""
    span, chord = p["wing_span_m"], p["wing_chord_m"]
    area = span * chord
    aspect = span / chord
    a2d = p["wing_cla_2d"]
    cla = a2d / (1.0 + a2d / (math.pi * OSWALD * aspect))
    alpha0 = math.radians(p["wing_alpha0_deg"] + p["wing_incidence_deg"])
    weight = p["mass_kg"] * G
    cl_level = cla * alpha0
    return {
        "wing_area_m2": area,
        "aspect_ratio": aspect,
        "wing_cla": cla,
        "wing_alpha0_rad": alpha0,
        "wing_stall_rad": alpha0 + math.radians(p["wing_stall_deg"]),
        "cl_level": cl_level,
        "weight_n": weight,
        "lift_thrust_to_weight": 4.0 * p["lift_thrust_max_n"] * p["air_density"] / RHO_REF / weight,
        # share of the weight the wing carries in level flight at the cruise speed, still air
        "wing_lift_fraction_cruise": 0.5 * p["air_density"] * p["cruise_mps"] ** 2 * area * cl_level / weight,
    }


def _rotor(name, pose, axis_pose=""):
    return f"""
    <link name='{name}'>
      <pose>{pose}</pose>
      <inertial>
        <mass>0.005</mass>
        <inertia><ixx>9.75e-07</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.000166704</iyy><iyz>0</iyz><izz>0.000167604</izz></inertia>
      </inertial>
    </link>
    <joint name='{name}_joint' type='revolute'>{axis_pose}
      <child>{name}</child>
      <parent>base_link</parent>
      <axis>
        <xyz>{'1 0 0' if axis_pose else '0 0 1'}</xyz>
        <limit><lower>-1e+16</lower><upper>1e+16</upper></limit>
      </axis>
    </joint>"""


def _motor(name, number, direction, thrust_max, max_radps, moment, tau_up, tau_down):
    return f"""
    <plugin filename="gz-sim-multicopter-motor-model-system" name="gz::sim::systems::MulticopterMotorModel">
      <jointName>{name}_joint</jointName>
      <linkName>{name}</linkName>
      <turningDirection>{direction}</turningDirection>
      <timeConstantUp>{tau_up}</timeConstantUp>
      <timeConstantDown>{tau_down}</timeConstantDown>
      <maxRotVelocity>{max_radps}</maxRotVelocity>
      <motorConstant>{thrust_max / max_radps ** 2:.6e}</motorConstant>
      <momentConstant>{moment}</momentConstant>
      <commandSubTopic>command/motor_speed</commandSubTopic>
      <motorNumber>{number}</motorNumber>
      <rotorDragCoefficient>0.000106428</rotorDragCoefficient>
      <rollingMomentCoefficient>1e-06</rollingMomentCoefficient>
      <rotorVelocitySlowdownSim>20</rotorVelocitySlowdownSim>
      <motorType>velocity</motorType>
    </plugin>"""


def _surface(a0, cla, cda, stall, cp, area, rho, upward="0 0 1"):
    """One lift/drag surface. Gazebo's plugin makes lift, drag and moment all proportional to
    (a0 + angle of attack), so a surface has no moment of its own: where the force acts (cp)
    is the only pitching-moment parameter."""
    return f"""
    <plugin filename="gz-sim-lift-drag-system" name="gz::sim::systems::LiftDrag">
      <a0>{a0:.6f}</a0>
      <cla>{cla:.6f}</cla>
      <cda>{cda:.6f}</cda>
      <cma>0.0</cma>
      <alpha_stall>{stall:.6f}</alpha_stall>
      <cla_stall>-3.85</cla_stall>
      <cda_stall>-0.9233984055</cda_stall>
      <cma_stall>0</cma_stall>
      <cp>{cp[0]:.4f} {cp[1]:.4f} {cp[2]:.4f}</cp>
      <area>{area:.5f}</area>
      <air_density>{rho}</air_density>
      <forward>1 0 0</forward>
      <upward>{upward}</upward>
      <link_name>base_link</link_name>
    </plugin>"""


def model_sdf(p):
    d = derived(p)
    k = p["inertia_scale"]
    rho = p["air_density"]
    thrust_scale = rho / RHO_REF
    rx, ry = p["rotor_x_m"], p["rotor_y_m"]
    mis = p["lift_mismatch_pct"] / 100.0

    # PX4 motor order of a quad-X: front right, rear left, front left, rear right.
    lift = [("rotor_0", rx, -ry, "ccw", 1.0 + mis), ("rotor_1", -rx, ry, "ccw", 1.0 - mis),
            ("rotor_2", rx, ry, "cw", 1.0), ("rotor_3", -rx, -ry, "cw", 1.0)]
    links, motors = "", ""
    for i, (name, x, y, direction, gain) in enumerate(lift):
        links += _rotor(name, f"{x} {y} 0.07 0 0 0")
        motors += _motor(name, i, direction, p["lift_thrust_max_n"] * gain * thrust_scale,
                         LIFT_MAX_RADPS, p["lift_moment_m"], p["lift_tau_up_s"], p["lift_tau_down_s"])
    # The pusher link is pitched 90 degrees so that its z axis, the thrust axis, points forward.
    links += _rotor("rotor_pusher",
                    f"{p['pusher_x_m']} {p['pusher_y_m']} {p['pusher_z_m']} 0 1.5707963 0",
                    "\n      <pose>0 0 0 0 -1.5707963 0</pose>")
    motors += _motor("rotor_pusher", 4, "cw", p["pusher_thrust_max_n"] * thrust_scale,
                     PUSHER_MAX_RADPS, p["pusher_moment_m"], 0.05, 0.05)

    # Drag is cda * (a0 + alpha) in the plugin; cda is set so that it equals cd0 in level flight.
    a0 = d["wing_alpha0_rad"]
    wing_cda = p["wing_cd0"] / max(a0, 0.02)
    quarter_span = p["wing_span_m"] / 4.0
    cp = (p["wing_cp_x_m"], quarter_span, p["wing_cp_z_m"])
    half = d["wing_area_m2"] / 2.0
    surfaces = _surface(a0, d["wing_cla"], wing_cda, d["wing_stall_rad"], cp, half, rho)
    surfaces += _surface(a0, d["wing_cla"], wing_cda, d["wing_stall_rad"],
                         (cp[0], -cp[1], cp[2]), half, rho)
    # The flat plate between the booms: a symmetric surface that stalls early.
    surfaces += _surface(math.radians(p["tail_incidence_deg"]), p["tail_cla"], 0.5, 0.2,
                         (-p["tail_arm_m"], 0.0, 0.0), p["tail_area_m2"], rho)
    # No fin. One side-force surface for the fuselage, booms and pusher in a side wind.
    surfaces += _surface(0.0, p["side_cla"], 0.5, 0.25, (p["side_cp_x_m"], 0.0, 0.0),
                         p["side_area_m2"], rho, upward="0 1 0")

    return f"""<?xml version="1.0"?>
<!-- Generated by year_end_flight/sim/model.py. Placeholder vehicle: see vehicle.yaml. -->
<sdf version='1.10'>
  <model name='{MODEL_NAME}'>
    <pose>0 0 0.2 0 0 0</pose>
    <link name='base_link'>
      <inertial>
        <pose>{p['cg_x_m']} {p['cg_y_m']} {p['cg_z_m']} 0 0 0</pose>
        <mass>{p['mass_kg']}</mass>
        <inertia>
          <ixx>{p['ixx_kgm2'] * k}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>{p['iyy_kgm2'] * k}</iyy><iyz>0</iyz>
          <izz>{p['izz_kgm2'] * k}</izz>
        </inertia>
      </inertial>
      <collision name='base_collision'>
        <pose>0 0 -0.07 0 0 0</pose>
        <geometry><box><size>0.7 0.55 0.05</size></box></geometry>
      </collision>{SENSORS}
    </link>{links}{surfaces}{motors}
    <static>0</static>
  </model>
</sdf>
"""


def world_sdf(p, stock_world, speed_factor):
    """PX4's default world at this simulation speed, with this case's steady wind added."""
    # The speed is written here as well as asked of the running world by PX4's start-up
    # script: with several simulations at once that request does not always arrive.
    stock_world = stock_world.replace("<real_time_factor>1.0</real_time_factor>",
                                      f"<real_time_factor>{speed_factor}</real_time_factor>")
    speed = p["wind_speed_mps"]
    if speed <= 0.0:
        return stock_world
    # The vehicle is spawned facing +x. wind_from_deg is measured from the nose toward the left.
    a = math.radians(p["wind_from_deg"])
    wind = f"""    <wind>
      <linear_velocity>{-speed * math.cos(a):.4f} {-speed * math.sin(a):.4f} 0</linear_velocity>
    </wind>
"""
    marker = "<spherical_coordinates>"
    if marker not in stock_world:
        raise ValueError("stock world has no <spherical_coordinates> to insert the wind before")
    head, tail = stock_world.split(marker, 1)
    return head.rstrip(" ") + wind.lstrip(" ") + "    " + marker + tail


def px4_params(cfg):
    """PX4 parameters of a case: the control concept plus the geometry PX4 is told."""
    n = cfg["nominal"]
    rx, ry = n["rotor_x_m"], n["rotor_y_m"]
    params = {
        "CA_ROTOR0_PX": rx, "CA_ROTOR0_PY": ry,
        "CA_ROTOR1_PX": -rx, "CA_ROTOR1_PY": -ry,
        "CA_ROTOR2_PX": rx, "CA_ROTOR2_PY": -ry,
        "CA_ROTOR3_PX": -rx, "CA_ROTOR3_PY": ry,
        "CA_ROTOR4_PX": n["pusher_x_m"],
    }
    params.update(cfg["px4_params"])
    return params


def write_case(case_dir, p, cfg):
    """Create the files PX4 and Gazebo read for one run. Returns the environment additions."""
    case_dir = Path(case_dir)
    px4_root = Path(cfg["sim"]["px4_root"]).expanduser()
    model_dir = case_dir / "models" / MODEL_NAME
    world_dir = case_dir / "worlds"
    model_dir.mkdir(parents=True, exist_ok=True)
    world_dir.mkdir(parents=True, exist_ok=True)

    (model_dir / "model.sdf").write_text(model_sdf(p), newline="\n")
    (model_dir / "model.config").write_text(
        f"<?xml version='1.0'?>\n<model>\n  <name>{MODEL_NAME}</name>\n  <version>1.0</version>\n"
        "  <sdf version='1.10'>model.sdf</sdf>\n  <description>placeholder quad + pusher</description>\n"
        "</model>\n", newline="\n")
    stock = (px4_root / "Tools/simulation/gz/worlds/default.sdf").read_text(encoding="utf-8-sig")
    (world_dir / "default.sdf").write_text(world_sdf(p, stock, cfg["sim"]["speed_factor"]),
                                           newline="\n")
    gusty = p["wind_speed_mps"] > 0.0 and p["wind_gust_pct"] > 0.0
    gusts = GUSTS.format(amplitude=p["wind_gust_pct"]) if gusty else ""
    (case_dir / "server.config").write_text(SERVER_CONFIG.format(gusts=gusts), newline="\n")
    # PX4's start-up script sources this file from its working directory before starting Gazebo.
    (case_dir / "gz_env.sh").write_text(
        f"export PX4_GZ_MODELS={case_dir}/models\n"
        f"export PX4_GZ_WORLDS={world_dir}\n"
        f"export GZ_SIM_RESOURCE_PATH={case_dir}/models:{world_dir}\n"
        f"export GZ_SIM_SERVER_CONFIG_PATH={case_dir}/server.config\n", newline="\n")

    env = {f"PX4_PARAM_{k}": str(v) for k, v in px4_params(cfg).items()}
    env.update({
        "HEADLESS": "1",
        "PX4_SYS_AUTOSTART": str(cfg["sim"]["autostart"]),
        "PX4_SIM_MODEL": f"gz_{MODEL_NAME}",
        "PX4_GZ_WORLD": "default",
        "PX4_SIM_SPEED_FACTOR": str(cfg["sim"]["speed_factor"]),
    })
    return env
