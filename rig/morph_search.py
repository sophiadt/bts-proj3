"""Search how far the lab head can be morphed toward a fit while the rig works.

The professor's procedure: start from the lab head (it works), morph a small
step toward the person's fitted head, check the rig still works, grow the step
while it does, and back off when it does not. Each step t (fraction of the way
from the lab head to the fit; pipeline/morph_steps.py) is checked by:

  1. pipeline/verify.py (no licence): mesh, folds, volume, penetration,
     attachments, mouth, teeth, eyes. A failing step never opens Maya.
  2. rig/write_rest.py --ui-ready: the step written into the lab scene.
  3. rig/sim_check.py: exact rest, same mesh and UVs, stable for frames 0-50.
  4. rig/ui_smoke_test.py --reuse-generic: "not even you" through the
     keyboard UI; finite, not too fast, mouth and lip muscle open.

Adaptive schedule: t = 0.25 first; after a pass the next increment is 1.5x
the last; after a fail, halfway between the last pass and the fail. Stops at
t = 1 passing, after MAX_STEPS checked steps, or when the pass/fail gap is
under MIN_GAP. The furthest passing step is kept as scenes/<id>_morph.mb and
_morph_ui.mb; the other steps' scenes are deleted (430 MB each).

Each Maya step runs as its own mayapy job, one after another: the Ziva batch
licence has one seat.

    mayapy rig/morph_search.py SUBJECT [--max-steps 6] [--start 0.25]
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from core import io as cio                             # noqa: E402
from pipeline import morph_steps                       # noqa: E402

MAX_STEPS = 6
START_T = 0.25
GROW = 1.5
MIN_GAP = 0.05
ROOT = Path(__file__).resolve().parents[1]


def run_job(args, log):
    """Run one mayapy script; -> (ok, exit code, seconds). Output to `log`."""
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        code = subprocess.call([sys.executable] + [str(a) for a in args], cwd=str(ROOT), stdout=fh,
                               stderr=subprocess.STDOUT)
    return code == 0, code, round(time.time() - t0, 1)


def check_step(subject, t, logdir, verbose=True):
    """All four checks for one step. -> dict(t, pass, failed_at, ...)."""
    out, rep = morph_steps.write_step(subject, t, verbose=False)
    name = out.name
    res = {"t": t, "subject": name, "verify": rep["verify"]["verdict"], "verify_failed": rep["verify"]["failed"]}
    if verbose:
        print("t = {0:.3f}: verify {1}{2}".format(t, rep["verify"]["verdict"].upper(),
                                                   " ({0})".format(", ".join(rep["verify"]["failed"]))
                                                   if rep["verify"]["failed"] else ""), flush=True)
    if rep["verify"]["verdict"] == "fail":
        return dict(res, **{"pass": False, "failed_at": "verify"})
    scenes = config.template_export_dir() / "scenes"
    jobs = [("write_rest", ["rig/write_rest.py", out, "--save", name, "--ui-ready"]),
            ("sim_check", ["rig/sim_check.py", scenes / (name + ".mb"), "--subject", out]),
            ("ui_smoke", ["rig/ui_smoke_test.py", name, "--reuse-generic"])]
    for label, args in jobs:
        ok, code, sec = run_job(args, logdir / "{0}_{1}.log".format(name, label))
        res[label] = {"ok": ok, "exit": code, "seconds": sec}
        if verbose:
            print("          {0}: {1} ({2:.0f} s)".format(label, "pass" if ok else "FAIL exit {0}".format(code), sec),
                  flush=True)
        if not ok:
            return dict(res, **{"pass": False, "failed_at": label})
    return dict(res, **{"pass": True, "failed_at": None})


def search(subject, max_steps=MAX_STEPS, start=START_T, verbose=True):
    with cio.RunDir(subject, "morph_search", "rig/morph_search.py") as run:
        logdir = run.path / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        steps, best, worst_fail = [], 0.0, None
        t, inc = start, start
        while len(steps) < max_steps:
            r = check_step(subject, t, logdir, verbose)
            steps.append(r)
            run.metrics["steps"] = steps
            if r["pass"]:
                best = t
                if t >= 1.0:
                    break
                inc *= GROW
                nxt = min(1.0, t + inc)
                if worst_fail is not None:
                    nxt = min(nxt, 0.5 * (t + worst_fail))
            else:
                worst_fail = t if worst_fail is None else min(worst_fail, t)
                nxt = 0.5 * (best + worst_fail)
            if worst_fail is not None and worst_fail - best < MIN_GAP:
                break
            t = round(nxt, 3)
        scenes = config.template_export_dir() / "scenes"
        keep = morph_steps.step_name(subject, best) if best > 0 else None
        if keep:
            for suffix in ("", "_ui"):
                src = scenes / (keep + suffix + ".mb")
                if src.exists():
                    shutil.copy2(src, scenes / (subject + "_morph" + suffix + ".mb"))
        for r in steps:                                  # 430 MB each: keep only the chosen step's copy
            for suffix in ("", "_ui"):
                p = scenes / (r["subject"] + suffix + ".mb")
                if p.exists():
                    p.unlink()
        run.metrics.update({"furthest_working_t": best, "first_failing_t": worst_fail,
                            "kept_scene": (subject + "_morph.mb") if keep else None, "kept_step": keep})
        if verbose:
            print("furthest working step: t = {0:.3f}{1}".format(
                best, "" if worst_fail is None else " (first failing: {0:.3f})".format(worst_fail)))
            if keep:
                print("kept as", scenes / (subject + "_morph.mb"))
        return run.metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subject", help="a fitted subject (from the lab head) under template_export/subjects")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--start", type=float, default=START_T)
    args = ap.parse_args()
    search(args.subject, args.max_steps, args.start)


if __name__ == "__main__":
    main()
