#!/usr/bin/env python3
"""Fly one simulation case and write its figures.

Starts PX4 SITL with a headless Gazebo, flies the profile of ../README.md in offboard mode over
MAVLink (the same position/velocity setpoints the ROS 2 nodes send), lands, and reads the PX4
log. Runs on Linux (WSL); needs pymavlink, pyulog and a built PX4 tree (vehicle.yaml, sim).

    python3 run_case.py                              # the nominal vehicle
    python3 run_case.py --set wind_speed_mps=6 --set wind_from_deg=90
"""
import argparse
import json
import math
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import yaml

os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402
import model  # noqa: E402

OFFBOARD = 6                    # PX4 custom main mode
AUTO, AUTO_LAND = 4, 6          # PX4 custom main and sub mode
# SET_POSITION_TARGET_LOCAL_NED type masks: a set bit means "ignore this field".
HOLD_POSITION = 8 | 16 | 32 | 64 | 128 | 256 | 2048             # x, y, z, yaw
FLY_VELOCITY = 1 | 2 | 32 | 64 | 128 | 256 | 2048               # vx, vy, z, yaw
DESCEND = 4 | 8 | 16 | 64 | 128 | 256 | 2048                    # x, y, vz, yaw
DESCENT_MPS = 1.5
LAND_FROM_M = 4.0
TURN_RATE = math.radians(20.0)  # how fast the commanded heading may change
TURN_TIME_S = 9.0               # extra hover to turn the nose into the wind


class Abort(Exception):
    pass


class Link:
    """MAVLink connection to one PX4 instance, and the latest state it reported."""

    def __init__(self, instance):
        self.conn = mavutil.mavlink_connection(f"udpin:0.0.0.0:{14540 + instance}",
                                               source_system=255, source_component=190)
        self.t = None               # seconds since PX4 boot, simulation time
        self.pos = None             # north, east, down [m]
        self.vel = None
        self.att = None             # roll, pitch, yaw [rad]
        self.armed = False
        self.main_mode = None
        self.texts = []
        self._last_send = 0.0
        self._last_heartbeat = 0.0
        self.setpoint = None        # (mask, x, y, z, vx, vy, vz, yaw)

    def pump(self, timeout=0.02):
        """Read one message and keep the setpoint stream alive (PX4 leaves offboard without it)."""
        msg = self.conn.recv_match(blocking=True, timeout=timeout)
        if msg is not None:
            kind = msg.get_type()
            if kind == "LOCAL_POSITION_NED":
                self.t = msg.time_boot_ms * 1e-3
                self.pos = (msg.x, msg.y, msg.z)
                self.vel = (msg.vx, msg.vy, msg.vz)
            elif kind == "ATTITUDE":
                self.att = (msg.roll, msg.pitch, msg.yaw)
            elif kind == "HEARTBEAT" and msg.get_srcComponent() == 1:
                self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self.main_mode = (msg.custom_mode >> 16) & 0xFF
            elif kind == "STATUSTEXT":
                self.texts.append(msg.text)
        now = time.monotonic()
        if self.setpoint is not None and now - self._last_send >= 0.01:
            mask, x, y, z, vx, vy, vz, yaw = self.setpoint
            self.conn.mav.set_position_target_local_ned_send(
                0, self.conn.target_system, self.conn.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED, mask, x, y, z, vx, vy, vz, 0, 0, 0, yaw, 0)
            self._last_send = now

        if now - self._last_heartbeat >= 0.5:       # PX4 counts this as its ground station
            self.conn.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                         mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            self._last_heartbeat = now
        return msg

    def command(self, command, *params):
        params = list(params) + [0] * (7 - len(params))
        self.conn.mav.command_long_send(self.conn.target_system, self.conn.target_component,
                                        command, 0, *params)

    def set_param(self, name, value):
        """Set one PX4 parameter and wait for PX4 to report the new value.

        A value given to PX4 at start-up that equals the firmware default is not kept: the
        airframe file's own default then wins. Setting it here, after boot, always holds."""
        target = (self.conn.target_system, self.conn.target_component)
        key = name.encode()

        def reply(send):
            deadline = time.monotonic() + 5.0
            send()
            while time.monotonic() < deadline:
                msg = self.pump(0.05)
                if msg is not None and msg.get_type() == "PARAM_VALUE" and msg.param_id == name:
                    return msg
            raise Abort(f"PX4 did not answer for parameter {name}")

        current = reply(lambda: self.conn.mav.param_request_read_send(*target, key, -1))
        is_int = current.param_type == mavutil.mavlink.MAV_PARAM_TYPE_INT32
        # PX4 carries an integer parameter as its four bytes inside the float field.
        raw = struct.unpack("<f", struct.pack("<i", int(value)))[0] if is_int else float(value)
        reply(lambda: self.conn.mav.param_set_send(*target, key, raw, current.param_type))


def fly(p, params, instance, wall_limit_s=420.0):
    """Fly the profile. Returns what happened, with the phase times metrics.compute() needs."""
    result = {"completed": False, "fail_reason": None, "phases": {}, "alt_m": p["alt_m"]}
    link = Link(instance)
    wall_start = time.monotonic()

    def spin(until, limit_s, what):
        """Pump until until() is true; limit_s is simulation time."""
        t0 = link.t
        while not until():
            link.pump()
            if time.monotonic() - wall_start > wall_limit_s:
                raise Abort(f"wall-clock limit during {what}")
            if t0 is None:
                t0 = link.t
            elif link.t - t0 > limit_s:
                raise Abort(f"timeout in {what}")
            if airborne[0]:
                if link.att and max(abs(link.att[0]), abs(link.att[1])) > math.radians(60):
                    raise Abort(f"tumbled in {what}")
                if not link.armed:
                    raise Abort(f"disarmed in {what}")
                if link.main_mode != OFFBOARD and what != "land":
                    raise Abort(f"left offboard in {what}")
                if home[2] - link.pos[2] < 1.0 and what != "land":
                    raise Abort(f"hit the ground in {what}")

    def phase(name, until, limit_s):
        start = link.t
        spin(until, limit_s, name)
        result["phases"][name] = [start, link.t]

    airborne, home = [False], [0.0, 0.0, 0.0]
    try:
        if link.conn.wait_heartbeat(timeout=90) is None:
            raise Abort("no heartbeat from PX4")
        while link.pos is None or link.att is None:
            link.pump(0.2)
            if time.monotonic() - wall_start > 120:
                raise Abort("no position estimate")
        for name, value in params.items():
            link.set_param(name, value)
        home[:] = link.pos
        yaw0 = link.att[2]
        result["home"], result["yaw0"] = list(home), yaw0
        alt, v_cruise, accel = p["alt_m"], p["cruise_mps"], p["accel_mps2"]
        cos0, sin0 = math.cos(yaw0), math.sin(yaw0)
        z_fly = home[2] - alt

        # Heading policy. 0: hold the first heading, which is the direction of travel.
        # 1: point the nose along the velocity relative to the air, as a weathervane would.
        # The true mean wind is used here; the vehicle will need an estimate of it.
        into_wind = p["heading_into_wind"] >= 0.5
        a = math.radians(p["wind_from_deg"])
        wind_n = -p["wind_speed_mps"] * (math.cos(a) * cos0 + math.sin(a) * sin0)
        wind_e = -p["wind_speed_mps"] * (math.cos(a) * sin0 - math.sin(a) * cos0)
        yaw_cmd, t_yaw = [yaw0], [link.t]

        def heading(speed):
            if not into_wind:
                return yaw0
            air_n, air_e = speed * cos0 - wind_n, speed * sin0 - wind_e
            # capped, so the first call after the climb cannot turn the heading in one step
            dt, t_yaw[0] = min(0.1, max(0.0, link.t - t_yaw[0])), link.t
            if math.hypot(air_n, air_e) > 1.0:
                error = math.atan2(air_e, air_n) - yaw_cmd[0]
                error = math.atan2(math.sin(error), math.cos(error))
                yaw_cmd[0] += max(-TURN_RATE * dt, min(TURN_RATE * dt, error))
            return yaw_cmd[0]

        # --- arm and take off: PX4 wants the setpoint stream before it accepts offboard -------
        link.setpoint = (HOLD_POSITION, home[0], home[1], z_fly, 0, 0, 0, yaw0)
        t_wait, t_ask = link.t, -1e9
        while not (link.armed and link.main_mode == OFFBOARD):
            link.pump()
            if link.t - t_wait > 60:
                raise Abort("could not arm in offboard: " + " | ".join(link.texts[-4:]))
            if link.t - t_wait > 1.5 and link.t - t_ask > 1.0:
                link.command(mavutil.mavlink.MAV_CMD_DO_SET_MODE, 1, OFFBOARD)
                link.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
                t_ask = link.t
        phase("climb", lambda: home[2] - link.pos[2] > alt - 0.5 and abs(link.vel[2]) < 0.3, 60)
        airborne[0] = True

        def hover_over(point):
            link.setpoint = (HOLD_POSITION, point[0], point[1], z_fly, 0, 0, 0, heading(0.0))

        t0 = link.t
        hover1_s = p["hover_s"] + (TURN_TIME_S if into_wind else 0.0)

        def hover1():
            hover_over(home)
            return link.t - t0 > hover1_s
        phase("hover1", hover1, hover1_s + 5)

        # --- "transition": the commanded speed rises at the acceleration limit ---------------
        def forward(speed):
            link.setpoint = (FLY_VELOCITY, 0, 0, z_fly, speed * cos0, speed * sin0, 0,
                             heading(speed))

        t0 = link.t

        def ramp_up():
            forward(min(v_cruise, accel * (link.t - t0)))
            return link.t - t0 > v_cruise / accel
        phase("accel", ramp_up, v_cruise / accel + 5)

        t0 = link.t
        phase("cruise", lambda: link.t - t0 > p["cruise_s"], p["cruise_s"] + 5)

        # --- "back transition": the commanded speed falls, then the vehicle holds a point ----
        t0 = link.t

        def ramp_down():
            speed = max(0.0, v_cruise - accel * (link.t - t0))
            forward(speed)
            return speed == 0.0 and math.hypot(link.vel[0], link.vel[1]) < 0.5
        phase("decel", ramp_down, v_cruise / accel + 25)

        hold = link.pos
        t0 = link.t

        def hover2():
            hover_over(hold)
            return link.t - t0 > p["hover_s"]
        phase("hover2", hover2, p["hover_s"] + 5)

        def descend():
            link.setpoint = (DESCEND, hold[0], hold[1], 0, 0, 0, DESCENT_MPS, heading(0.0))
            return home[2] - link.pos[2] < LAND_FROM_M
        phase("descend", descend, alt / DESCENT_MPS + 20)

        airborne[0] = False
        link.command(mavutil.mavlink.MAV_CMD_DO_SET_MODE, 1, AUTO, AUTO_LAND)
        phase("land", lambda: not link.armed, 60)
        result["touchdown_err_m"] = math.hypot(link.pos[0] - hold[0], link.pos[1] - hold[1])
        result["completed"] = True
    except Abort as stop:
        result["fail_reason"] = str(stop)
    result["texts"] = link.texts[-12:]
    time.sleep(1.0)                 # let the logger close the file after the disarm
    return result


def launch(case_dir, env_add, cfg, instance):
    build = Path(cfg["sim"]["px4_root"]).expanduser() / "build/px4_sitl_default"
    env = dict(os.environ, **env_add)
    env["GZ_PARTITION"] = f"psi_qp_{instance}_{os.getpid()}"      # keeps other Gazebos out
    log = open(case_dir / "px4.log", "w")
    return subprocess.Popen([str(build / "bin/px4"), "-i", str(instance), "-d", str(build / "etc")],
                            cwd=case_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def stop(proc, case_dir):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            continue
    # Gazebo's server is started by a launcher and can outlive the group; its command line
    # names this case's world file.
    subprocess.run(["pkill", "-9", "-f", str(case_dir / "worlds")], check=False)


def run(cfg, overrides, case_id, instance):
    p = dict(cfg["nominal"])
    for name, value in overrides.items():
        if name not in p:
            raise KeyError(f"unknown vehicle value: {name}")
        p[name] = value
    case_dir = Path(cfg["sim"]["run_root"]) / case_id
    shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True)

    proc = launch(case_dir, model.write_case(case_dir, p, cfg), cfg, instance)
    try:
        flight = fly(p, model.px4_params(cfg), instance)
    finally:
        stop(proc, case_dir)
    # PX4's own warnings, for a run that did not go as planned
    seen = [line for line in (case_dir / "px4.log").read_text(errors="replace").splitlines()
            if line.startswith(("WARN", "ERROR")) and "Resolve system health" not in line]
    flight["px4_warnings"] = list(dict.fromkeys(seen))[-15:]

    figures = {}
    logs = sorted(case_dir.glob("log/*/*.ulg"))
    if logs:
        try:
            figures = metrics.compute(logs[-1], flight)
        except Exception as error:      # a log cut short by a crash must not lose the result
            flight["log_error"] = repr(error)
    passed, reasons = metrics.verdict(flight, figures)
    record = {"case": case_id, "passed": passed, "reasons": reasons, "overrides": overrides,
              "derived": model.derived(p), "flight": flight, "figures": figures,
              "log": str(logs[-1]) if logs else None}
    (case_dir / "result.json").write_text(json.dumps(record, indent=1))
    return record


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=str(here / "vehicle.yaml"))
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="override one vehicle value; may be repeated")
    parser.add_argument("--px4", action="append", default=[], metavar="NAME=VALUE",
                        help="override one PX4 parameter; may be repeated")
    parser.add_argument("--case", default="single")
    parser.add_argument("--instance", type=int, default=5,
                        help="PX4 instance number; sets the MAVLink port (14540 + instance)")
    parser.add_argument("--json", help="also write the record to this file")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["px4_params"].update({k: float(v) for k, v in (item.split("=", 1) for item in args.px4)})
    overrides = {k: float(v) for k, v in (item.split("=", 1) for item in args.set)}
    record = run(cfg, overrides, args.case, args.instance)
    if args.json:
        Path(args.json).write_text(json.dumps(record, indent=1))
    print(json.dumps({k: record[k] for k in ("case", "passed", "reasons", "figures")}, indent=1))
    print("derived:", json.dumps(record["derived"]))
    print("phases:", json.dumps(record["flight"]["phases"]))
    for line in record["flight"]["px4_warnings"]:
        print("px4:", line)
    print("log:", record["log"])


if __name__ == "__main__":
    main()
