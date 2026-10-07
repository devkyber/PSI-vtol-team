"""Figures of one run, read from the PX4 log (ULog)."""
import math

import numpy as np
from pyulog import ULog

# A run passes when it completes and stays inside all of these.
LIMITS = {
    "tilt_max_deg": 30.0,           # largest roll or pitch while airborne
    "alt_err_max_m": 3.0,           # largest height error between take-off and descent
    "lift_sat_frac": 0.05,          # share of forward flight with a lift motor at its maximum
    "torque_unachieved_frac": 0.10  # share of forward flight where the allocator fell short
}

TOPICS = ["vehicle_attitude", "vehicle_attitude_setpoint", "vehicle_local_position",
          "actuator_motors", "rate_ctrl_status", "control_allocator_status", "airspeed_validated"]


def _topic(ulog, name, multi_id=0):
    for d in ulog.data_list:
        if d.name == name and d.multi_id == multi_id:
            return d.data
    return None


def _window(data, span):
    t = data["timestamp"] * 1e-6
    return (t >= span[0]) & (t <= span[1])


def _euler_deg(att, field="q"):
    w, x, y, z = (att[f"{field}[{i}]"] for i in range(4))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def compute(ulog_path, flight):
    """flight: what run_case.fly() returned (phases in seconds since PX4 boot, home, yaw0)."""
    ulog = ULog(str(ulog_path), message_name_filter_list=TOPICS)
    ph = flight["phases"]
    out = {}

    def span(first, last):
        if first in ph and last in ph:
            return (ph[first][0], ph[last][1])
        return None

    airborne = span("hover1", "hover2") or span("hover1", "cruise") or span("hover1", "hover1")
    forward = span("accel", "decel") or span("accel", "accel")
    cruise = ph.get("cruise")

    att = _topic(ulog, "vehicle_attitude")
    pos = _topic(ulog, "vehicle_local_position")
    mot = _topic(ulog, "actuator_motors")
    if att is None or pos is None or mot is None or airborne is None:
        return out
    roll, pitch, yaw = _euler_deg(att)

    m = _window(att, airborne)
    if m.any():
        out["tilt_max_deg"] = float(max(np.abs(roll[m]).max(), np.abs(pitch[m]).max()))
    m = _window(pos, airborne)
    if m.any():
        out["alt_err_max_m"] = float(np.abs(pos["z"][m] - (flight["home"][2] - flight["alt_m"])).max())

    lift = np.stack([mot[f"control[{i}]"] for i in range(4)])
    pusher = np.nan_to_num(mot["control[4]"])      # PX4 logs a stopped motor as NaN
    m = _window(mot, ph["hover1"])
    if m.any():
        out["hover_lift_cmd"] = float(lift[:, m].mean())

    if forward:
        m = _window(att, forward)
        if m.any():
            out["fwd_roll_max_deg"] = float(np.abs(roll[m]).max())
            out["fwd_pitch_max_deg"] = float(np.abs(pitch[m]).max())
            # heading error against the heading PX4 was asked to hold at that moment
            asked = _topic(ulog, "vehicle_attitude_setpoint")
            if asked is not None:
                yaw_sp = np.unwrap(np.radians(_euler_deg(asked, "q_d")[2]))
                yaw_sp = np.degrees(np.interp(att["timestamp"][m], asked["timestamp"], yaw_sp))
                out["fwd_yaw_err_max_deg"] = float(np.abs(_wrap_deg(yaw[m] - yaw_sp)).max())
        m = _window(pos, forward)
        if m.any():
            # distance to the left or right of the line flown, from home along the first heading
            dn, de = pos["x"][m] - flight["home"][0], pos["y"][m] - flight["home"][1]
            cross = -dn * math.sin(flight["yaw0"]) + de * math.cos(flight["yaw0"])
            out["cross_track_max_m"] = float(np.abs(cross).max())
        m = _window(mot, forward)
        if m.any():
            out["pusher_max_cmd"] = float(pusher[m].max())
            out["pusher_sat_frac"] = float((pusher[m] >= 0.88).mean())
            out["lift_max_cmd"] = float(lift[:, m].max())
            out["lift_min_cmd"] = float(lift[:, m].min())
            out["lift_sat_frac"] = float((lift[:, m].max(axis=0) >= 0.98).mean())
        rate = _topic(ulog, "rate_ctrl_status")
        if rate is not None:
            m = _window(rate, forward)
            if m.any():
                for axis in ("roll", "pitch", "yaw"):
                    out[f"{axis}_integ_max"] = float(np.abs(rate[f"{axis}speed_integ"][m]).max())
        alloc = _topic(ulog, "control_allocator_status")
        if alloc is not None:
            m = _window(alloc, forward)
            if m.any():
                out["torque_unachieved_frac"] = float(1.0 - alloc["torque_setpoint_achieved"][m].mean())

    if cruise:
        late = (max(cruise[0], cruise[1] - 5.0), cruise[1])     # after the speed has settled
        m = _window(pos, late)
        if m.any():
            ground = np.hypot(pos["vx"][m], pos["vy"][m])
            out["cruise_speed_mps"] = float(ground.mean())
        m = _window(att, late)
        if m.any():
            out["cruise_pitch_deg"] = float(pitch[m].mean())
        m = _window(mot, late)
        if m.any():
            out["cruise_pusher_cmd"] = float(pusher[m].mean())
            out["cruise_lift_cmd"] = float(lift[:, m].mean())
        air = _topic(ulog, "airspeed_validated")
        if air is not None:
            m = _window(air, late)
            if m.any():
                out["cruise_airspeed_mps"] = float(np.nanmean(air["true_airspeed_m_s"][m]))
    return out


def verdict(flight, figures):
    """(passed, reasons)."""
    reasons = []
    if not flight["completed"]:
        reasons.append(flight["fail_reason"] or "not completed")
    for name, limit in LIMITS.items():
        if name not in figures:
            reasons.append(f"{name} missing")
        elif figures[name] > limit:
            reasons.append(f"{name} {figures[name]:.2f} > {limit}")
    return (not reasons, reasons)
