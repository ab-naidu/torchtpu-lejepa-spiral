"""Summarize matched spiral ablations and measured TPU timing across training seeds."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
import json
import os
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from report_results import cumulative, observed_crossover, pair_key


SEEDS = (1337, 1338, 1339)
MODES = ("eager", "fused", "compile")
NAMES = {"eager": "Strict eager", "fused": "Defer and Fuse", "compile": "Compile"}
COLORS = {"eager": "#273951", "fused": "#B65E1A", "compile": "#087E84",
          "cpu": "#667389", "tpu": "#087E84"}
WEIGHTS = (0.0, 0.001)


def science_recipe(run):
    """Allow comparisons across objective, seed and device but preserve the numerical recipe."""
    cfg = dict(run["config"])
    for field in ("weight", "data_seed"):
        cfg.pop(field)
    return json.dumps([cfg, run["metadata"]["source_sha256"]], sort_keys=True)


def matched(records, device, steps, ablation=False):
    groups = defaultdict(dict)
    expected = WEIGHTS if ablation else MODES
    for path, run in records:
        cfg = run["config"]
        if (run["device"] != device or cfg["steps"] != steps or run["seed"] not in SEEDS
                or run["status"] != "ok" or not (run.get("parity") or {}).get("passed")):
            continue
        if ablation and (cfg["weight"] not in WEIGHTS or run["mode"] != "eager"):
            continue
        if not ablation and (cfg["weight"] != 0.001 or run["mode"] not in MODES):
            continue
        # All settings, including the group on the allocation and framework versions,
        # must agree; the two science runs intentionally differ in one loss weight.
        comparable = dict(run, config=dict(cfg, weight=0) if ablation else cfg)
        key = pair_key(comparable)
        if key is None:
            continue
        member = cfg["weight"] if ablation else run["mode"]
        if member in groups[key]:
            raise ValueError(f"Duplicate {member} in comparison group at {path}")
        groups[key][member] = (path, run)

    result = {}
    signatures = set()
    for key, group in groups.items():
        if set(group) != set(expected):
            continue
        seed, cfg, meta = json.loads(key)
        if seed in result:
            raise ValueError(f"Multiple eligible {steps}-step {device} groups for seed {seed}; select one results tree")
        cfg.pop("data_seed")
        meta.pop("comparison_group")
        signatures.add(json.dumps([cfg, meta], sort_keys=True))
        result[seed] = group
    if len(signatures) > 1:
        raise ValueError(f"Incompatible hardware, software or training settings across {device} {steps}-step seeds")
    return dict(sorted(result.items()))


def metric(group, weight, name):
    return group[weight][1]["metrics"][name]


def format_stats(values, spec):
    center = format(mean(values), spec)
    return center if len(values) == 1 else f"{center} ± {format(stdev(values), spec)}"


def source_links(groups, out):
    links = []
    for seed, group in groups.items():
        refs = []
        for member, (path, _) in group.items():
            relative = quote(Path(os.path.relpath(path, out)).as_posix())
            label = NAMES.get(member, f"λ={member}")
            refs.append(f"[{label}]({relative})")
        links.append(f"{seed} ({', '.join(refs)})")
    return " · ".join(links)


def science_figure(tpu, cpu, output):
    seed = 1337 if 1337 in tpu else next(iter(tpu))
    scatter = tpu[seed][0.001][1]["scatter"]
    true = np.asarray(scatter["true"])
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 7.1), layout="constrained")
    for ax, field, title in zip(axes[0], ("observed", "recovered"),
                                ("Observed spiral", "Decoded coordinates with SIGReg")):
        points = np.asarray(scatter[field])
        dots = ax.scatter(*points.T, c=true[:, 0], cmap="viridis", s=10, linewidths=0)
        ax.set(aspect="equal", title=title, xlabel="Coordinate 1", ylabel="Coordinate 2")
    fig.colorbar(dots, ax=axes[0, :], label="True coordinate 1", shrink=0.8)

    for ax, field, label in zip(axes[1], ("r2", "total_variance"),
                                ("Held-out linear-probe R²", "Embedding covariance trace")):
        for device, pairs, shift, marker in (("cpu", cpu, -0.06, "s"), ("tpu", tpu, 0.06, "o")):
            for i, (s, group) in enumerate(pairs.items()):
                x = np.array([0, 1]) + shift + (SEEDS.index(s) - 1) * 0.018
                y = [metric(group, w, field) for w in WEIGHTS]
                ax.plot(x, y, color=COLORS[device], lw=0.9, alpha=0.58)
                ax.scatter(x, y, s=33, facecolor="white" if device == "cpu" else COLORS[device],
                           edgecolor=COLORS[device], marker=marker, label=device.upper() if i == 0 else None, zorder=3)
        ax.set(xlim=(-0.26, 1.26), xticks=[0, 1], xticklabels=["Alignment only", "+ SIGReg"], ylabel=label)
        if field == "r2":
            raw = metric(tpu[seed], 0.001, "raw_r2")
            ax.axhline(raw, ls=":", color="#59616B", lw=1, label="Raw linear probe")
            ax.legend(fontsize=8, loc="upper left")
        else:
            # Zero is an actual possible collapse outcome; a symlog scale can display it.
            vals = [metric(g, w, field) for pairs in (tpu, cpu) for g in pairs.values() for w in WEIGHTS]
            if min(vals) > 0:
                ax.set_yscale("log")
            elif any(v > 0 for v in vals):
                ax.set_yscale("symlog", linthresh=min(v for v in vals if v > 0) / 10)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"Spiral recovery after 6,000 updates\nHeld-out coordinates from the TPU-trained encoder · seed {seed}", fontsize=14)
    fig.savefig(output, dpi=195, bbox_inches="tight")
    plt.close(fig)


def timing_section(groups, steps, out):
    first_run = next(iter(groups.values()))["eager"][1]
    meta, cfg = first_run["metadata"], first_run["config"]
    lines = ["## When does compilation pay back its first-use cost?", "",
             f"This comparison covers {steps:,} updates per mode at training seeds {', '.join(map(str, groups))}, "
             f"with {cfg['block_size']}-step timing blocks after the first update. "
             f"The runtime reported {meta['device_count']} visible {meta['runtime_device_name']} "
             f"{'device' if meta['device_count'] == 1 else 'devices'}. The fixed-shape {cfg['dtype']} encoder uses "
             f"width {cfg['width']}, batch size {cfg['batch_size']} and {cfg['slices']} projections. "
             f"Python {meta['python']}, PyTorch {meta['torch_version']}, TorchTPU {meta['torch_tpu_version']}, "
             f"libtpu {meta['libtpu_version']}.", ""]
    fig, ax = plt.subplots(figsize=(8.4, 4.5), layout="constrained")
    boundaries = None
    table = ["| Execution mode | First update (s) | Warm updates/s | Final cumulative time (s) |",
             "| --- | ---: | ---: | ---: |"]
    for mode in MODES:
        runs = [group[mode][1] for group in groups.values()]
        traces = [cumulative(r) for r in runs]
        current = list(traces[0])
        if (current[-1] != steps or any(list(trace) != current for trace in traces)
                or (boundaries is not None and boundaries != current)):
            raise ValueError("TPU timing records have inconsistent measured boundaries")
        boundaries = current
        matrix = np.array([list(trace.values()) for trace in traces])
        color = COLORS[mode]
        if len(runs) > 1:
            ax.fill_between(current, matrix.min(axis=0), matrix.max(axis=0), color=color, alpha=0.14, linewidth=0)
        ax.plot(current, np.median(matrix, axis=0), color=color, label=NAMES[mode], lw=2)
        table.append(f"| {NAMES[mode]} | {median(r['timing']['first_step_seconds'] for r in runs):.2f} | "
                     f"{median(r['timing']['steady_steps_per_second'] for r in runs):.1f} | "
                     f"{median(trace[steps] for trace in traces):.2f} |")
    subtitle = (f"Median and observed range across {len(groups)} matched training seeds"
                if len(groups) > 1 else f"Single matched training seed {next(iter(groups))}")
    ax.set(xlabel="Completed optimizer updates", ylabel="Cumulative training time (s)", xlim=(0, steps),
           title=f"One {meta['runtime_device_name']} device · {steps:,} updates per mode\n{subtitle}")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(out / "timing.png", dpi=195, bbox_inches="tight")
    plt.close(fig)
    caption = (f"Lines show the pointwise median and shading spans the {len(groups)} observed seeds."
               if len(groups) > 1 else "Lines show one measured seed. More seeds are needed to estimate run-to-run spread.")
    totals = {m: median(cumulative(g[m][1])[steps] for g in groups.values()) for m in MODES}
    best = min(totals, key=totals.get)
    lines.extend([f"At update {steps:,}, **{NAMES[best]}** had the lowest median cumulative time "
                  f"({totals[best]:.2f} s). Strict eager took {totals['eager']:.2f} s and compilation took "
                  f"{totals['compile']:.2f} s.", "", "![Measured training time by execution mode](timing.png)", "", caption,
                  "", *table, "", "Table entries are medians across the included seeds. "
                  "Warm throughput excludes the first update and first subsequent block.", "",
                  "| Seed | Defer and Fuse vs eager | Compile vs eager |", "| ---: | --- | --- |"])
    for seed, group in groups.items():
        base = cumulative(group["eager"][1])
        result = [observed_crossover(base, cumulative(group[m][1])) for m in ("fused", "compile")]
        lines.append(f"| {seed} | {' | '.join(result)} |")
    lines.extend(["", "Crossovers are bounded by synchronized observations. Forward, backward, AdamW and dispatch "
                  "are timed. Input preparation, copies, evaluation and parity checks are outside the timer. "
                  "Compilation captures the forward objective. The first update includes execution and startup.", "",
                  "Timing records — " + source_links(groups, out)])
    return lines


def summarize(root, out):
    records = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in sorted(root.rglob("result.json"))]
    if not records:
        raise ValueError(f"No result.json under {root}")
    out.mkdir(exist_ok=True, parents=True)
    cpu = matched(records, "cpu", 6000, ablation=True)
    tpu = matched(records, "tpu", 6000, ablation=True)
    recipes = {science_recipe(g[0.001][1]) for pairs in (cpu, tpu) for g in pairs.values()}
    if len(recipes) > 1:
        raise ValueError("CPU and TPU ablation settings or source hashes differ")
    lines = ["# Spiral recovery and the cost of compiling on TorchTPU", "",
             "Two experiments separate what the encoder learns from how fast the training loop runs. "
             "Each included run passed a one-step comparison of its loss and all parameter gradients with CPU.",
             "", "## Does the encoder recover the hidden coordinates?", ""]
    if tpu:
        science_figure(tpu, cpu, out / "science.png")
        avg_r2 = [mean(metric(g, w, "r2") for g in tpu.values()) for w in WEIGHTS]
        lines.extend([f"Adding SIGReg changed mean TPU recovery from **R² {avg_r2[0]:.3f}** to "
                      f"**{avg_r2[1]:.3f}** after 6,000 updates on the same training seeds.", "",
                      "![Held-out spiral recovery and seed-paired ablations](science.png)", "",
                      "Point color identifies the original first coordinate. The bottom panels connect the "
                      "two objectives within each training seed.", ""])
    else:
        lines.extend(["No complete TPU science pair is available for the selected seeds.", ""])
    if cpu or tpu:
        lines.extend(["| Device | Training seeds | Objective | Held-out R² | Embedding covariance trace |",
                      "| --- | --- | --- | ---: | ---: |"])
        for device, pairs in (("TPU", tpu), ("CPU", cpu)):
            for w in WEIGHTS:
                if pairs:
                    r2 = format_stats([metric(g, w, "r2") for g in pairs.values()], ".3f")
                    trace = format_stats([metric(g, w, "total_variance") for g in pairs.values()], ".4g")
                    objective = "Alignment only" if w == 0 else "Alignment + SIGReg"
                    lines.append(f"| {device} | {', '.join(map(str, pairs))} | {objective} | {r2} | {trace} |")
        example = next(iter((tpu or cpu).values()))[0.001][1]
        c = example["config"]
        lines.extend(["", f"Values are means across training seeds with sample SD when multiple seeds are available. "
                      f"Each model trained for {c['steps']:,} updates. The affine probe uses {c['probe_fit']:,} "
                      f"fit examples and {c['probe_test']:,} held-out examples shared across seeds "
                      f"(evaluation seed {c['eval_seed']}). The raw-input linear probe scored "
                      f"{example['metrics']['raw_r2']:.5f}. The covariance trace sums variance along both learned "
                      "dimensions. Values near zero indicate nearly constant outputs.", ""])
        if tpu:
            lines.extend(["| TPU seed | R², alignment | R², + SIGReg | Trace, alignment | Trace, + SIGReg |",
                          "| ---: | ---: | ---: | ---: | ---: |"])
            for seed, group in tpu.items():
                lines.append(f"| {seed} | {metric(group, 0, 'r2'):.3f} | {metric(group, 0.001, 'r2'):.3f} | "
                             f"{metric(group, 0, 'total_variance'):.4g} | {metric(group, 0.001, 'total_variance'):.4g} |")
            lines.append("")
        for name, pairs in (("TPU", tpu), ("CPU", cpu)):
            if pairs:
                lines.extend([name + " ablation records — " + source_links(pairs, out), ""])
    for steps in (1001, 201):
        groups = matched(records, "tpu", steps)
        if groups:
            lines.extend(timing_section(groups, steps, out))
            break
    else:
        lines.extend(["## TPU execution time", "", "No complete matched set of TPU timing modes is available."])
    report = out / "summary.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(summarize(args.results, args.out))
