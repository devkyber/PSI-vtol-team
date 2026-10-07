#!/usr/bin/env python3
"""Run many simulation cases and tabulate them.

    python3 sweep.py oat                    # each swept value at its low and high end, alone
    python3 sweep.py wind                   # steady wind by speed and direction
    python3 sweep.py random --n 40          # all swept values drawn at once, uniformly

Ranges come from vehicle.yaml (sweep). Results go to out/<study>.csv and out/<study>.json;
each case's files and PX4 log stay under the run_root of vehicle.yaml.
"""
import argparse
import csv
import json
import queue
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
COLUMNS = ["tilt_max_deg", "alt_err_max_m", "fwd_roll_max_deg", "fwd_pitch_max_deg",
           "fwd_yaw_err_max_deg", "cross_track_max_m", "cruise_speed_mps", "cruise_pusher_cmd",
           "pusher_max_cmd", "pusher_sat_frac", "hover_lift_cmd", "cruise_lift_cmd",
           "lift_max_cmd", "lift_min_cmd", "lift_sat_frac", "roll_integ_max", "pitch_integ_max",
           "yaw_integ_max", "torque_unachieved_frac"]


def oat_cases(cfg):
    """The nominal vehicle, then one value at a time at each end of its range."""
    cases = [("nominal", {})]
    top_wind = cfg["sweep"]["wind_speed_mps"][1]
    for name, (low, high) in cfg["sweep"].items():
        if name == "wind_from_deg":
            continue
        if name == "wind_speed_mps":        # a wind needs a direction to mean anything
            for direction in (0, 90, 180, 270):
                cases.append((f"wind{top_wind:g}_from{direction}",
                              {"wind_speed_mps": top_wind, "wind_from_deg": float(direction)}))
            continue
        for label, value in (("lo", low), ("hi", high)):
            if value != cfg["nominal"][name]:
                extra = {"wind_speed_mps": 5.0} if name == "wind_gust_pct" else {}
                cases.append((f"{name}_{label}", {name: float(value), **extra}))
    return cases


# Heading holds compared by the wind study. "firm" is the nominal one of vehicle.yaml: PX4's
# own yaw gains. "soft" lets the heading give way to whatever turns the nose in a side wind.
POLICIES = {"firm": {},
            "soft": {"px4:MC_YAW_P": 0.3, "px4:MC_YAWRATE_I": 0.0}}


def wind_cases():
    """Steady wind by speed and direction, under each heading hold; the rest nominal."""
    return [(f"wind{speed}_from{direction}_{label}",
             {"wind_speed_mps": float(speed), "wind_from_deg": float(direction), **policy})
            for speed in (2, 4, 6, 8) for direction in (0, 45, 90, 135, 180)
            for label, policy in POLICIES.items()]


def random_cases(cfg, count, seed):
    rng = random.Random(seed)
    return [(f"rand{seed}_{i:03d}", {name: rng.uniform(low, high)
                                     for name, (low, high) in cfg["sweep"].items()})
            for i in range(count)]


def run_one(case, overrides, common, config, instances, study):
    instance = instances.get()
    try:
        out = HERE / "out" / study / f"{case}.json"
        cmd = [sys.executable, str(HERE / "run_case.py"), "--config", config, "--case",
               f"{study}/{case}", "--instance", str(instance), "--json", str(out)]
        for name, value in {**common, **overrides}.items():
            if name.startswith("px4:"):         # a PX4 parameter rather than a vehicle value
                cmd += ["--px4", f"{name[4:]}={value}"]
            else:
                cmd += ["--set", f"{name}={value}"]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if not out.exists():
            return {"case": case, "passed": False, "overrides": overrides, "figures": {},
                    "reasons": ["runner failed: " + done.stderr.strip()[-300:]]}
        record = json.loads(out.read_text())
        record["case"] = case
        return record
    finally:
        instances.put(instance)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("study", choices=["oat", "wind", "random"])
    parser.add_argument("--config", default=str(HERE / "vehicle.yaml"))
    parser.add_argument("--n", type=int, default=40, help="number of random cases")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="a vehicle value for every case of the study; may be repeated")
    parser.add_argument("--name", help="name of the output files; default: the study")
    parser.add_argument("--jobs", type=int, default=3, help="cases flown at the same time")
    parser.add_argument("--first-instance", type=int, default=5)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cases = {"oat": lambda: oat_cases(cfg), "wind": wind_cases,
             "random": lambda: random_cases(cfg, args.n, args.seed)}[args.study]()
    study = args.name or (f"random_seed{args.seed}" if args.study == "random" else args.study)
    common = {k: float(v) for k, v in (item.split("=", 1) for item in args.set)}
    (HERE / "out" / study).mkdir(parents=True, exist_ok=True)

    instances = queue.Queue()
    for i in range(args.jobs):
        instances.put(args.first_instance + i)
    records = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, case, overrides, common, args.config, instances, study)
                   for case, overrides in cases]
        for future in futures:
            record = future.result()
            records.append(record)
            f = record["figures"]
            print(f"{record['case']:28s} {'pass' if record['passed'] else 'FAIL'}"
                  f"  tilt {f.get('tilt_max_deg', float('nan')):5.1f} deg"
                  f"  alt {f.get('alt_err_max_m', float('nan')):4.2f} m"
                  f"  pusher {f.get('pusher_max_cmd', float('nan')):4.2f}"
                  f"  lift {f.get('lift_max_cmd', float('nan')):4.2f}"
                  f"  {'; '.join(record['reasons'])}", flush=True)

    (HERE / "out" / f"{study}.json").write_text(json.dumps(records, indent=1))
    with open(HERE / "out" / f"{study}.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "passed", "reasons", "overrides"] + COLUMNS)
        for r in records:
            writer.writerow([r["case"], int(r["passed"]), "; ".join(r["reasons"]),
                             json.dumps(r["overrides"])]
                            + [r["figures"].get(c, "") for c in COLUMNS])
    passed = sum(r["passed"] for r in records)
    print(f"\n{passed} of {len(records)} passed; table in out/{study}.csv")


if __name__ == "__main__":
    main()
