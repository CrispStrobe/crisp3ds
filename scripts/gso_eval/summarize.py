"""One table row per case in work/: registered views, camera error, F1, times."""
import json
import os
import re
import sys
from pathlib import Path

here = Path(os.environ.get("GSO_EVAL_DIR", Path(__file__).resolve().parents[2] / ".local-tools/gso-eval"))
cases = sys.argv[1:] or sorted(p.name for p in (here / "work").iterdir() if p.is_dir())
print("| case | photos | registered | cam centre % / orient deg | F1 0.5/1/2 % | acc med/p90 % | comp med/p90 % | render s | run s |")
for name in cases:
    w = here / "work" / name
    log = (w / "log.txt").read_text() if (w / "log.txt").exists() else ""
    times = re.search(r"render_s=(\d+) run_s=(\d+) run_status=(\d+)", log)
    fe = w / "keep/run_frontend_frontend.json"
    fe = fe if fe.exists() else w / "run/frontend/frontend.json"
    gates = (json.loads(fe.read_text()).get("gates") or {}) if fe.exists() else {}
    truth = w / "capture/truth.json"
    photos = len(json.loads(truth.read_text())["views"]) if truth.exists() else "?"
    cam = json.loads((w / "camera_error.json").read_text()) if (w / "camera_error.json").exists() else None
    res = w / "score/result.json"
    if res.exists():
        m = json.loads(res.read_text())["metrics"]["all"]
        f1 = " / ".join(f"{v['f1']:.3f}" for v in m["thresholds"].values())
        a = m["accuracy_mesh_to_reference"]["fraction_of_diagonal"]
        c = m["completeness_reference_to_mesh"]["fraction_of_diagonal"]
        acc = f"{100 * a['median']:.2f} / {100 * a['p90']:.2f}"
        comp = f"{100 * c['median']:.2f} / {100 * c['p90']:.2f}"
    else:
        reason = re.search(r"failed: (.*)", log)
        f1, acc, comp = ("refused: " + reason.group(1)[:90]) if reason else "-", "", ""
    camtxt = f"{cam['proper']:.2f} / {cam['orientation_deg_median']:.2f}" if cam else ""
    reg = gates.get("registered_views", "-")
    print(f"| {name} | {photos} | {reg} | {camtxt} | {f1} | {acc} | {comp} | "
          f"{times.group(1) if times else ''} | {times.group(2) if times else ''} |")
