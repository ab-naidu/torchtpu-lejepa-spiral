"""Summarize measured LeJEPA runs and plot their held-out predictions and timings."""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def scatter_figure(run, output):
    data = run["scatter"]
    true = np.asarray(data["true"])
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), layout="constrained")
    for ax, key, title in zip(axes, ("true", "observed", "recovered"),
                              ("True coordinates", "Observed spiral", "Decoded coordinates")):
        points = np.asarray(data[key])
        dots = ax.scatter(points[:, 0], points[:, 1], c=true[:, 0], cmap="viridis",
                          s=9, alpha=0.85, linewidths=0)
        ax.set(title=title, aspect="equal", xlabel="Coordinate 1")
    extent = np.vstack((true, np.asarray(data["recovered"])))
    low, high = float(extent.min()), float(extent.max())
    pad = max(0.05, 0.05 * (high - low))
    for ax in (axes[0], axes[2]):
        ax.set(xlim=(low - pad, high + pad), ylim=(low - pad, high + pad))
    axes[0].set_ylabel("Coordinate 2")
    fig.colorbar(dots, ax=axes, shrink=0.65, label="True coordinate 1")
    fig.suptitle(f"{run['device'].upper()} {run['mode']} · weight {run['config']['weight']:g}"
                 f" · seed {run['seed']} · held-out R² {run['metrics']['r2']:.3f}")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def cumulative(run):
    timing = run["timing"]
    times = {1: timing["first_step_seconds"]}
    for interval in timing["warm_intervals"]:
        times[interval["end_step"]] = interval["cumulative_seconds"]
    return times


def observed_crossover(baseline, candidate):
    """Require the same measured boundaries; report final sustained lead and its uncertainty."""
    if baseline.keys() != candidate.keys():
        return "Different timing boundaries prevent a paired comparison."
    steps = sorted(baseline)
    wins = [candidate[s] <= baseline[s] for s in steps]
    if not wins[-1]:
        return (f"Behind at step {steps[-1]}; an earlier observed lead reversed." if any(wins)
                else f"No observed break-even through step {steps[-1]}.")
    first = 1 + max((i for i, win in enumerate(wins) if not win), default=-1)
    if first == 0:
        return f"At or ahead from step {steps[0]} at all measured boundaries."
    return (f"Behind at step {steps[first-1]}, at or ahead by step {steps[first]}. "
            + (f"Still at or ahead at the sampled boundaries through step {steps[-1]}."
               if first < len(steps) - 1 else "That was the final measurement."))


def timing_figure(by_mode, output):
    fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
    for mode in ("eager", "fused", "compile"):
        if mode in by_mode:
            times = cumulative(by_mode[mode])
            ax.plot(list(times), list(times.values()), marker="o", markersize=4, label=mode)
    ax.set(xlabel="Completed training steps", ylabel="Cumulative training time (s)",
           title="TPU training time including the first step")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def pair_key(run):
    meta = run["metadata"]
    if not meta.get("comparison_group") or meta.get("device_label") in (None, "", "unreported"):
        return None
    fields = ("comparison_group", "device_label", "device_count", "topology",
              "runtime_device_name", "platform", "python", "torch_version", "torch_tpu_version",
              "libtpu_version", "cache", "source_sha256")
    conditions = [run["seed"], run["config"], {k: meta.get(k) for k in fields}]
    return json.dumps(conditions, sort_keys=True)


def display_number(value, precision=3):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{value:.{precision}f}"
    return "—"


def write_report(runs_dir, out):
    files = sorted(runs_dir.rglob("result.json"))
    if not files:
        raise ValueError(f"No result.json found under {runs_dir}")
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# LeJEPA measurements", "",
             "Each row is one invocation. The first step includes initialization triggered by that step. "
             "The steady rate excludes step one and the first post-startup block. "
             "Covariance eigenvalues show the variance along both learned dimensions.", "",
             "| Run | Device / mode | Seed | SIGReg weight | Status | Raw input R² | Learned R² | "
             "Covariance eigenvalues | First step (s) | Steady steps/s |",
             "|---|---|---:|---:|---|---:|---:|---|---:|---:|"]
    successful = []
    groups = defaultdict(list)
    issues = []
    for path in files:
        run = json.loads(path.read_text(encoding="utf-8"))
        label = path.relative_to(runs_dir).parent.as_posix().replace("|", "&#124;")
        timing, metrics = run.get("timing") or {}, run.get("metrics") or {}
        status = run["status"]
        if status == "ok" and not (run.get("parity") or {}).get("passed"):
            status = "parity unverified"
        weight = run.get("config", {}).get("weight")
        eigenvalues = metrics.get("covariance_eigenvalues") or []
        spectrum = ", ".join(f"{v:.3g}" if v is not None else "—" for v in eigenvalues) or "—"
        try:
            record_link = quote(Path(os.path.relpath(path, start=out)).as_posix())
        except ValueError:  # On Windows, a relative path cannot span drives.
            record_link = path.resolve().as_uri()
        cells = [f"[{label}]({record_link})", f"{run['device']} / {run['mode']}", run.get("seed", "—"),
                 weight, status, display_number(metrics.get("raw_r2")), display_number(metrics.get("r2")),
                 spectrum,
                 display_number(timing.get("first_step_seconds")), display_number(timing.get("steady_steps_per_second"), 1)]
        lines.append("| " + " | ".join(str(v) for v in cells) + " |")
        if status != "ok":
            detail = run.get("error") or status
            issues.append(f"- **{label}** {detail}")
            continue
        successful.append((label, run))
        if run["device"] == "tpu":
            key = pair_key(run)
            if key is not None:
                groups[key].append((label, run))

    if issues:
        lines.extend(["", "## Incomplete runs", "", *issues])

    if successful:
        lines.extend(["", "## Held-out coordinates", "",
                      "Color follows the same held-out sample across panels. A linear probe fitted on a separate split "
                      "maps learned coordinates into the true coordinate system. The table also reports the raw-input "
                      "linear probe. Compare variance with recovery to diagnose collapse."])
    for i, (label, run) in enumerate(successful, start=1):
        filename = f"coordinates-{i}.png"
        scatter_figure(run, out / filename)
        lines.extend(["", f"**{label}**", "", f"![Held-out coordinates]({filename})"])

    lines.extend(["", "## Measured TPU break-even", ""])
    pairs = 0
    for members in groups.values():
        by_mode = {run["mode"]: run for _, run in members}
        if len(by_mode) != len(members):
            lines.append("Duplicate modes in one matched group; use a unique comparison group for each repeat.")
            continue
        if "eager" not in by_mode or len(by_mode) < 2:
            continue
        eager = by_mode["eager"]
        filename = f"tpu-training-{pairs + 1}.png"
        timing_figure(by_mode, out / filename)
        pairs += 1
        lines.extend([f"Matched runs: {', '.join(label for label, _ in members)}.",
                      "", f"![Cumulative TPU training time]({filename})", ""])
        for mode, candidate in by_mode.items():
            if mode != "eager":
                lines.append(f"- **{mode} vs eager** — {observed_crossover(cumulative(eager), cumulative(candidate))}")
        lines.append("")
    if not pairs:
        lines.append("No matched, parity-passing TPU eager/comparison pair is available.")
    else:
        lines.append("Each point is a synchronized timing boundary. Cumulative time includes the first call. "
                     "Input staging outside timed blocks, process setup and evaluation are excluded. "
                     "Lines connect sampled points; the exact crossover between observations is unknown.")
    report = out / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(write_report(args.runs, args.out))
