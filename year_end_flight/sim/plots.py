#!/usr/bin/env python3
"""Draw the figures of the feasibility study from the sweep results.

    python3 plots.py            # writes ../figures/*.png and *.pdf

Reads out/oat.json, out/wind.json and out/random_seed1.json, and the PX4 log of the nominal case
for the time history. Needs matplotlib and pyulog.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics  # noqa: E402
import model  # noqa: E402

OUT = HERE / "out"
FIGURES = HERE.parent / "figures"

# Ink and surfaces
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS, BAND = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#f0efec"
# Two categorical hues, and one blue ramp for magnitude and for ordered series
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
LIGHT, MID, DARK = RAMP[3], RAMP[7], RAMP[11]
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", RAMP)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.fontsize": 8, "legend.handlelength": 1.6,
    "savefig.dpi": 220, "pdf.fonttype": 42,
})


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", FIGURES / f"{name}.png")


def load(name):
    return json.loads((OUT / f"{name}.json").read_text())


# ---------------------------------------------------------------------------------------------
# 0. The vehicle from above: what can be controlled
# ---------------------------------------------------------------------------------------------
PROP_RADIUS = 0.19              # drawn for illustration; the propeller size is not given


def vehicle_plan():
    from matplotlib.patches import Circle, FancyArrowPatch, Patch, Rectangle
    n = yaml.safe_load((HERE / "vehicle.yaml").read_text())["nominal"]
    rx, ry = n["rotor_x_m"], n["rotor_y_m"]
    span, chord = n["wing_span_m"], n["wing_chord_m"]
    plate_te = rx - 1.065       # trailing edge of the rear plate, 1065 mm behind the front motors
    fixed = dict(facecolor=GRID, edgecolor=AXIS, lw=1.0)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.set_aspect("equal")
    ax.axis("off")
    # nose to the right: x forward, y to the left of the vehicle is up on the page
    ax.add_patch(Rectangle((-chord / 2, -span / 2), chord, span, **fixed))          # wing
    ax.add_patch(Rectangle((plate_te, -0.2375), 0.15, 0.475, **fixed))              # rear plate
    for y in (ry, -ry):
        ax.plot([plate_te + 0.075, rx], [y, y], color=AXIS, lw=3, solid_capstyle="butt")
    ax.add_patch(Rectangle((-0.355, -0.1025), 0.71, 0.205, facecolor="#d6d5cd",
                           edgecolor=AXIS, lw=1.0))                                  # body
    # PX4 motor order of a quad-X and the spin of each
    rotors = [(1, rx, -ry, "CCW"), (2, -rx, ry, "CCW"), (3, rx, ry, "CW"), (4, -rx, -ry, "CW")]
    for number, x, y, spin in rotors:
        ax.add_patch(Circle((x, y), PROP_RADIUS, facecolor=BLUE, alpha=0.12, lw=0))
        ax.add_patch(Circle((x, y), PROP_RADIUS, fill=False, edgecolor=BLUE, lw=1.6))
        ax.plot(x, y, "o", ms=5, color=BLUE)
        ax.text(x, y + (0.075 if y > 0 else -0.075), f"{number}  {spin}", ha="center",
                va="center", fontsize=8.5, color=INK)
    px = n["pusher_x_m"]
    ax.plot([px, px], [-0.13, 0.13], color=ORANGE, lw=3.5, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((px + 0.03, 0), (px + 0.2, 0), arrowstyle="-|>",
                                 mutation_scale=12, color=ORANGE, lw=1.6))
    ax.plot(0, 0, "+", ms=11, mew=1.6, color=INK)
    ax.text(0.035, -0.05, "CG", fontsize=8, color=INK)

    def dimension(a, b, text, offset=(0, 0), **kw):
        ax.annotate("", a, b, arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8,
                                              shrinkA=0, shrinkB=0))
        ax.text((a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1], text,
                fontsize=7.5, color=INK2, ha="center", va="center", **kw)

    dimension((0.72, -span / 2), (0.72, span / 2), "wing span 1700", (0.035, 0), rotation=90)
    dimension((-chord / 2, 0.80), (chord / 2, 0.80), "chord 250", (0, 0.035))
    dimension((-rx, -0.52), (rx, -0.52), "motors 665", (0, -0.035))
    dimension((0.60, -ry), (0.60, ry), "505", (0.035, 0), rotation=90)
    dimension((plate_te, 0.52), (rx, 0.52), "front motors to the plate's trailing edge 1065",
              (0, 0.035))
    ax.text(plate_te + 0.075, -0.30, "rear plate\n475 x 150", fontsize=7.5, color=INK2,
            ha="center", va="top", linespacing=1.1)
    ax.text(0.17, 0.06, "body 710 x 205", fontsize=7.5, color=INK2, ha="center", va="center")
    ax.text(px - 0.03, 0.17, "pusher", fontsize=8.5, color=INK, ha="center")
    ax.annotate("", (0.98, -0.78), (0.80, -0.78),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0, mutation_scale=10))
    ax.text(0.89, -0.74, "forward", fontsize=8, color=INK2, ha="center")

    keys = [Line2D([], [], marker="o", ls="", ms=9, mfc="#dbe8f8", mec=BLUE, mew=1.6,
                   label="Lift rotor: height, roll, pitch, yaw"),
            Line2D([], [], color=ORANGE, lw=3.5, label="Pusher: forward force"),
            Patch(facecolor=GRID, edgecolor=AXIS, label="Fixed: no control surfaces")]
    ax.legend(handles=keys, loc="lower left", bbox_to_anchor=(0.0, 0.02), ncol=1,
              handlelength=1.8, labelspacing=0.6)
    ax.text(0.0, -0.01, "Lengths in mm, from the airframe team. Assumed: the CG at the centre "
            "of the four motors, the wing and the body\ncentred on it, the pusher behind the body. "
            "Propellers drawn for illustration.", transform=ax.transAxes, fontsize=7.5,
            color=INK2, va="top", linespacing=1.3)
    ax.set_xlim(-0.95, 1.02)
    ax.set_ylim(-0.92, 0.92)
    ax.set_title("The vehicle from above: five things can be commanded", pad=4)
    save(fig, "fig0_vehicle")


# ---------------------------------------------------------------------------------------------
# 1. One flight of the nominal vehicle
# ---------------------------------------------------------------------------------------------
PHASES = [("climb", "Climb"), ("hover1", "Hover"), ("accel", "Transition"), ("cruise", "Cruise"),
          ("decel", "Back-\ntransition"), ("hover2", "Hover"), ("descend", "Descent"),
          ("land", "Landing")]


def nominal_flight():
    from pyulog import ULog
    record = next(r for r in load("oat") if r["case"] == "nominal")
    flight = record["flight"]
    ulog = ULog(record["log"])

    def topic(name):
        return next(d.data for d in ulog.data_list if d.name == name and d.multi_id == 0)

    t0 = flight["phases"]["climb"][0]
    t1 = flight["phases"]["land"][1]
    att, pos = topic("vehicle_attitude"), topic("vehicle_local_position")
    asked, mot = topic("vehicle_local_position_setpoint"), topic("actuator_motors")
    cos0, sin0 = math.cos(flight["yaw0"]), math.sin(flight["yaw0"])

    def clip(data):
        t = data["timestamp"] * 1e-6
        keep = (t >= t0) & (t <= t1)
        return t[keep] - t0, keep

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.4), sharex=True,
                             gridspec_kw={"hspace": 0.34})
    for ax in axes:
        ax.grid(axis="x", visible=False)        # the phase bands already mark time
        for i, (key, _) in enumerate(PHASES):
            if i % 2 == 0:
                a, b = flight["phases"][key]
                ax.axvspan(a - t0, b - t0, color=BAND, lw=0)
    for key, label in PHASES:
        a, b = flight["phases"][key]
        axes[0].text((a + b) / 2 - t0, 1.25, label, transform=axes[0].get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=7.5, color=INK2, linespacing=0.95)

    ax = axes[0]
    t, keep = clip(asked)
    speed_asked = np.nan_to_num(asked["vx"][keep]) * cos0 + np.nan_to_num(asked["vy"][keep]) * sin0
    ax.plot(t, speed_asked, color=MUTED, label="Commanded")
    t, keep = clip(pos)
    ax.plot(t, pos["vx"][keep] * cos0 + pos["vy"][keep] * sin0, color=BLUE, label="Flown")
    ax.set_title("Speed along the track (m/s)")
    ax.set_ylim(-1, 12)
    ax.legend(loc="upper right", ncol=2)

    ax = axes[1]
    roll, pitch, _ = metrics._euler_deg(att)
    t, keep = clip(att)
    ax.plot(t, pitch[keep], color=BLUE, label="Pitch, nose up positive")
    ax.plot(t, roll[keep], color=ORANGE, label="Roll")
    ax.set_title("Attitude (degrees)")
    ax.set_ylim(-4, 8)
    ax.legend(loc="upper left", ncol=2)

    ax = axes[2]
    t, keep = clip(pos)
    ax.axhline(flight["alt_m"], color=MUTED, lw=1.0)
    ax.text(0.5, flight["alt_m"] + 0.9, "target", color=INK2, fontsize=7.5)
    ax.plot(t, flight["home"][2] - pos["z"][keep], color=BLUE)
    ax.set_title("Height (m)")
    ax.set_ylim(0, 25)

    ax = axes[3]
    t, keep = clip(mot)
    lift = np.stack([mot[f"control[{i}]"][keep] for i in range(4)])
    ax.fill_between(t, lift.min(axis=0), lift.max(axis=0), color=BLUE, alpha=0.12, lw=0)
    ax.plot(t, lift.mean(axis=0), color=BLUE, label="Lift rotors, mean and range")
    ax.plot(t, np.nan_to_num(mot["control[4]"][keep]), color=ORANGE, label="Pusher")
    ax.set_title("Motor command (0 to 1)")
    ax.set_ylim(0, 0.8)
    ax.legend(loc="upper right", ncol=2)
    ax.set_xlabel("Time from the start of the climb (s)")
    ax.set_xlim(0, t1 - t0)
    fig.suptitle("Nominal vehicle: one flight through the mission profile, no wind",
                 x=0.125, y=0.975, ha="left", fontsize=10.5, fontweight="bold")
    save(fig, "fig1_nominal_flight")


# ---------------------------------------------------------------------------------------------
# 2. Wind, by speed and direction, for the two heading holds
# ---------------------------------------------------------------------------------------------
def wind_envelope():
    records = {r["case"]: r["figures"] for r in load("wind")}
    speeds, directions = [2, 4, 6, 8], [0, 45, 90, 135, 180]
    rows = [("fwd_yaw_err_max_deg", "Heading off the commanded one, largest (degrees)", "{:.0f}"),
            ("cross_track_max_m", "Distance off the track, largest (m)", "{:.1f}"),
            ("lift_max_cmd", "Largest lift-motor command (0 to 1)", "{:.2f}")]
    holds = [("firm", "Firm heading hold (PX4's yaw gains)"),
             ("soft", "Soft heading hold (weak yaw gain, no integrator)")]

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.6),
                             gridspec_kw={"hspace": 0.62, "wspace": 0.10})
    for i, (key, title, fmt) in enumerate(rows):
        grids = [np.array([[records[f"wind{s}_from{d}_{hold}"][key] for d in directions]
                           for s in speeds]) for hold, _ in holds]
        top = max(g.max() for g in grids)
        low = 0.0 if key != "lift_max_cmd" else 0.4
        for j, (grid, (_, hold_title)) in enumerate(zip(grids, holds)):
            ax = axes[i, j]
            ax.grid(False)
            mesh = ax.pcolormesh(grid, cmap=SEQUENTIAL, vmin=low, vmax=top,
                                 edgecolors=SURFACE, linewidth=2)
            for (y, x), value in np.ndenumerate(grid):
                dark = (value - low) / (top - low) > 0.5
                ax.text(x + 0.5, y + 0.5, fmt.format(value), ha="center", va="center",
                        fontsize=7.5, color="#ffffff" if dark else INK)
            ax.set_xticks(np.arange(5) + 0.5,
                          ["0\nhead", "45", "90\ncross", "135", "180\ntail"], fontsize=7.5)
            ax.set_yticks(np.arange(4) + 0.5, [str(s) for s in speeds] if j == 0 else [])
            for side in ax.spines.values():
                side.set_visible(False)
            if j == 0:
                ax.set_ylabel("Wind (m/s)")
            if i == 2:
                ax.set_xlabel("Wind from (degrees off the nose)")
            ax.set_title(title if j == 0 else "", pad=6)
            if i == 0:
                ax.text(0.5, 1.28, hold_title, transform=ax.transAxes, ha="center",
                        fontsize=8.5, color=INK2)
        bar = fig.colorbar(mesh, ax=axes[i, :], fraction=0.03, pad=0.02)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=7.5)
    fig.suptitle("Steady wind: 10 m/s cruise, the other values nominal",
                 x=0.105, y=1.0, ha="left", fontsize=10.5, fontweight="bold")
    save(fig, "fig2_wind")


# ---------------------------------------------------------------------------------------------
# 3. One value at a time
# ---------------------------------------------------------------------------------------------
LABELS = {
    "mass_kg": "Mass", "inertia_scale": "Inertia, factor", "cg_x_m": "CG fore and aft",
    "cg_y_m": "CG sideways", "cg_z_m": "CG up and down", "rotor_x_m": "Lift motor fore-aft position",
    "lift_thrust_max_n": "Lift motor maximum thrust", "lift_moment_m": "Lift rotor reaction torque",
    "lift_tau_up_s": "Lift motor time constant", "lift_mismatch_pct": "Lift motor mismatch",
    "pusher_thrust_max_n": "Pusher maximum thrust", "pusher_y_m": "Pusher thrust line sideways",
    "pusher_z_m": "Pusher thrust line up and down", "wing_chord_m": "Wing chord",
    "wing_incidence_deg": "Wing incidence", "wing_cd0": "Drag coefficient",
    "wing_cp_x_m": "Where the wing lift acts", "tail_area_m2": "Rear plate area",
    "tail_arm_m": "Rear plate distance", "tail_incidence_deg": "Rear plate incidence",
    "side_area_m2": "Side area", "side_cp_x_m": "Where the side force acts",
    "air_density": "Air density", "cruise_mps": "Cruise speed", "accel_mps2": "Acceleration limit",
}
UNITS = {"mass_kg": "kg", "cg_x_m": "m", "cg_y_m": "m", "cg_z_m": "m", "rotor_x_m": "m",
         "lift_thrust_max_n": "N", "lift_moment_m": "m", "lift_tau_up_s": "s",
         "lift_mismatch_pct": "%", "pusher_thrust_max_n": "N", "pusher_y_m": "m", "pusher_z_m": "m",
         "wing_chord_m": "m", "wing_incidence_deg": "deg", "wing_cp_x_m": "m", "tail_area_m2": "m²",
         "tail_arm_m": "m", "tail_incidence_deg": "deg", "side_area_m2": "m²", "side_cp_x_m": "m",
         "air_density": "kg/m³", "cruise_mps": "m/s", "accel_mps2": "m/s²"}


def one_at_a_time():
    cfg = yaml.safe_load((HERE / "vehicle.yaml").read_text())
    records = {r["case"]: r["figures"] for r in load("oat")}
    nominal = records["nominal"]
    columns = [("lift_max_cmd", "Largest lift-motor\ncommand (0 to 1)"),
               ("pusher_max_cmd", "Largest pusher\ncommand (0 to 1)"),
               ("pitch_integ_max", "Pitch-rate integrator,\nlargest (limit 0.30)"),
               ("tilt_max_deg", "Largest roll or pitch\n(degrees)")]
    names = [n for n in LABELS if n in cfg["sweep"]]

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 6.4), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    rows = np.arange(len(names))[::-1]
    for ax, (key, title) in zip(axes, columns):
        ax.axvline(nominal[key], color=MUTED, lw=1.0, zorder=1)
        for y, name in zip(rows, names):
            ends = {end: records.get(f"{name}_{end}", nominal)[key] for end in ("lo", "hi")}
            ax.plot([ends["lo"], ends["hi"]], [y, y], color=AXIS, lw=1.2, zorder=2)
            ax.plot(ends["lo"], y, "o", ms=6.5, color=LIGHT, mec=SURFACE, mew=1.2, zorder=3)
            ax.plot(ends["hi"], y, "o", ms=6.5, color=DARK, mec=SURFACE, mew=1.2, zorder=4)
        ax.set_title(title, fontsize=8, fontweight="normal", color=INK2, pad=8)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.7, len(names) - 0.3)
        ax.margins(x=0.12)
        ax.tick_params(axis="x", labelsize=7.5)

    def row_label(name):
        low, high = cfg["sweep"][name]
        unit = UNITS.get(name, "")
        return f"{LABELS[name]}, {low:g} to {high:g}{' ' + unit if unit else ''}"

    axes[0].set_yticks(rows, [row_label(n) for n in names], fontsize=7.6)
    keys = [Line2D([], [], marker="o", ls="", ms=6.5, color=LIGHT, label="Low end of the range"),
            Line2D([], [], marker="o", ls="", ms=6.5, color=DARK, label="High end of the range"),
            Line2D([], [], color=MUTED, lw=1.0, label="Nominal vehicle")]
    fig.legend(handles=keys, loc="lower left", bbox_to_anchor=(0.125, 0.015), ncol=3)
    fig.suptitle("One value changed at a time, across its range", x=-0.17, y=0.99, ha="left",
                 fontsize=10.5, fontweight="bold")
    save(fig, "fig3_one_at_a_time")


# ---------------------------------------------------------------------------------------------
# 4. Everything changed at once
# ---------------------------------------------------------------------------------------------
def wing_share(r):
    """Share of the weight the wing carries at the airspeed of the cruise, wind included."""
    o = r["overrides"]
    a = math.radians(o["wind_from_deg"])
    airspeed = math.hypot(o["cruise_mps"] + o["wind_speed_mps"] * math.cos(a),
                          o["wind_speed_mps"] * math.sin(a))
    return r["derived"]["wing_lift_fraction_cruise"] * (airspeed / o["cruise_mps"]) ** 2


def random_study():
    flown = [r for r in load("random_seed1") if r["figures"]]
    cfg = yaml.safe_load((HERE / "vehicle.yaml").read_text())

    def crosswind(r):
        o = r["overrides"]
        return abs(o["wind_speed_mps"] * math.sin(math.radians(o["wind_from_deg"])))

    panels = [(wing_share, "Wing's share of the weight\nat the airspeed flown",
               "torque_unachieved_frac", "Share of forward flight with the\nallocator short of torque",
               metrics.LIMITS["torque_unachieved_frac"]),
              (lambda r: r["derived"]["lift_thrust_to_weight"], "Lift thrust-to-weight\nratio",
               "lift_sat_frac", "Share of forward flight with a\nlift motor at its maximum",
               metrics.LIMITS["lift_sat_frac"]),
              (crosswind, "Crosswind component\n(m/s)",
               "fwd_yaw_err_max_deg", "Heading off the commanded one,\nlargest (degrees)", None)]
    inside = [r for r in flown if r["passed"]]
    outside = [r for r in flown if not r["passed"]]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.42})
    for ax, (x_of, x_label, key, y_label, limit) in zip(axes, panels):
        if limit is not None:
            ax.axhline(limit, color=MUTED, lw=1.0)
            ax.annotate("limit", (1.0, limit), xycoords=("axes fraction", "data"),
                        xytext=(-2, 3), textcoords="offset points", ha="right",
                        fontsize=7.5, color=INK2)
        ax.plot([x_of(r) for r in inside], [r["figures"][key] for r in inside], "o", ms=6,
                color=BLUE, mec=SURFACE, mew=1.2)
        ax.plot([x_of(r) for r in outside], [r["figures"][key] for r in outside], "X", ms=7.5,
                color=ORANGE, mec=SURFACE, mew=1.0)
        ax.set_xlabel(x_label)
        ax.set_title(y_label, fontsize=8, fontweight="normal", color=INK2, pad=8)
        ax.set_ylim(bottom=-0.03 * ax.get_ylim()[1])
    keys = [Line2D([], [], marker="o", ls="", ms=6, color=BLUE,
                   label=f"Inside every limit ({len(inside)})"),
            Line2D([], [], marker="X", ls="", ms=7.5, color=ORANGE,
                   label=f"Outside at least one limit ({len(outside)})")]
    fig.legend(handles=keys, loc="lower left", bbox_to_anchor=(0.06, -0.24), ncol=2)
    fig.suptitle(f"All {len(cfg['sweep'])} values drawn at random: {len(flown)} flights",
                 x=0.06, y=1.13, ha="left", fontsize=10.5, fontweight="bold")
    save(fig, "fig4_random")


# ---------------------------------------------------------------------------------------------
# 5. How much of the weight the wing carries
# ---------------------------------------------------------------------------------------------
def wing_share_curve():
    cfg = yaml.safe_load((HERE / "vehicle.yaml").read_text())
    airspeed = np.linspace(4, 22, 200)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    n = cfg["nominal"]
    for incidence, color in zip((0.0, 2.0, 4.0), (LIGHT, MID, DARK)):
        p = dict(n, wing_incidence_deg=incidence)
        d = model.derived(p)
        share = d["wing_lift_fraction_cruise"] * (airspeed / p["cruise_mps"]) ** 2
        ax.plot(airspeed, share, color=color, label=f"{incidence:g}°")
        ax.annotate(f"{incidence:g}°", (airspeed[-1], min(share[-1], 1.17)), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=INK2,
                    annotation_clip=False)
    ax.axhline(0.4, color=MUTED, lw=1.0)
    ax.text(4.3, 0.425, "40 %: proposed cap", fontsize=7.5, color=INK2)
    ax.set_ylim(0, 1.2)
    ax.set_xlim(4, 22)
    ax.set_xlabel("Airspeed (m/s)")
    ax.set_title("Wing's share of the weight in level flight", pad=20)
    ax.text(0.0, 1.03, f"{n['mass_kg']:g} kg, NACA 4415 wing of {n['wing_span_m']:g} m by "
            f"{n['wing_chord_m']:g} m", transform=ax.transAxes, fontsize=8, color=INK2)
    ax.legend(loc="upper left", title="Wing incidence", title_fontsize=8)
    save(fig, "fig5_wing_share")


if __name__ == "__main__":
    vehicle_plan()
    if "--vehicle-only" in sys.argv:
        sys.exit()
    nominal_flight()
    wind_envelope()
    one_at_a_time()
    random_study()
    wing_share_curve()
