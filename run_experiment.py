"""Run one Spiral LeJEPA configuration in a fresh process and save its measurements."""

import argparse
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback


class UnsupportedMode(RuntimeError):
    pass


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "tpu"], default="cpu")
    parser.add_argument("--mode", choices=["eager", "fused", "compile"], default="eager")
    parser.add_argument("--output", type=Path, required=True, help="New directory for this run")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--slices", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight", type=float, default=0.001, help="SIGReg weight")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--block-size", type=int, default=10, help="Steps per timed block")
    parser.add_argument("--threads", type=int, default=1, help="CPU intra-op threads")
    parser.add_argument("--device-label", default="", help="Observed TPU chip type, e.g. v6e-1")
    parser.add_argument("--comparison-group", default="", help="Same identifier for runs on the same allocation")
    args = parser.parse_args()
    if min(args.steps, args.width, args.slices, args.block_size, args.threads) < 1 or args.batch_size < 2:
        parser.error("steps, width, slices, block size and threads must be positive; batch size must be at least 2")
    if not (0 <= args.weight <= 1 and 0 < args.rho < 1 and math.isfinite(args.lr) and args.lr > 0
            and math.isfinite(args.weight_decay) and args.weight_decay >= 0 and 0 <= args.seed < 2**63 - 1):
        parser.error("invalid regularization, optimizer, correlation or seed")
    return args


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def setup_report(args):
    source = hashlib.sha256()
    for name in ("run_experiment.py", "spiral.py"):
        source.update(name.encode())
        source.update(Path(__file__).with_name(name).read_bytes())
    return {
        "schema_version": 1, "status": "running", "device": args.device, "mode": args.mode, "seed": args.seed,
        "config": {
            "steps": args.steps, "batch_size": args.batch_size, "width": args.width, "slices": args.slices,
            "weight": args.weight, "lr": args.lr, "weight_decay": args.weight_decay, "rho": args.rho,
            "block_size": args.block_size, "threads": args.threads, "dtype": "float32",
            "optimizer": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "schedule": "constant",
            "data_seed": args.seed + 1, "eval_seed": 1729, "probe_fit": 2048, "probe_test": 2048,
            "scatter_samples": 512, "parity_rtol": 0.02, "parity_atol": 1e-4,
        },
        "metadata": {
            "source_sha256": source.hexdigest(), "python": platform.python_version(),
            "platform": platform.platform(), "device_label": args.device_label or ("CPU" if args.device == "cpu" else "unreported"),
            "device_label_source": "user" if args.device_label else "default",
            "comparison_group": args.comparison_group,
            "torch_version": package_version("torch"), "torch_tpu_version": package_version("torch_tpu"),
            "libtpu_version": package_version("libtpu"), "topology": os.environ.get("TORCH_TPU_TOPOLOGY"),
            "cache": {
                "tier2": os.environ.get("TORCH_TPU_TIER2_COMPILATION_CACHE") if args.device == "tpu" else None,
                "tier3_configured": bool(os.environ.get("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT")) if args.device == "tpu" else False,
            },
        },
    }


def check_parity(torch, reference, target, reference_fn, target_fn, inputs, device, config):
    reference.zero_grad(set_to_none=True)
    target.zero_grad(set_to_none=True)
    expected = reference_fn(reference, *inputs, config["weight"])
    actual = target_fn(target, *(x.to(device) for x in inputs), config["weight"])
    expected.backward()
    actual.backward()
    expected, actual = expected.detach().cpu(), actual.detach().cpu()
    tolerances = {"rtol": config["parity_rtol"], "atol": config["parity_atol"]}
    loss_ok = torch.allclose(expected, actual, **tolerances)
    grad_checks = []
    for (name, cpu_param), target_param in zip(reference.named_parameters(), target.parameters()):
        if cpu_param.grad is None or target_param.grad is None:
            raise RuntimeError(f"Missing expected gradient for {name}")
        a, b = cpu_param.grad, target_param.grad.detach().cpu()
        grad_checks.append({"name": name, "max_abs_error": (a - b).abs().max().item(),
                            "passed": torch.allclose(a, b, **tolerances)})
    return {
        "passed": loss_ok and all(row["passed"] for row in grad_checks), "loss_passed": loss_ok,
        "reference_loss": expected.item(), "target_loss": actual.item(),
        "loss_abs_error": (expected - actual).abs().item(), "gradients": grad_checks, **tolerances,
    }


def run(args, report):
    if args.device == "cpu" and args.mode != "eager":
        raise UnsupportedMode(f"{args.mode} is a TPU execution mode; CPU supports the eager reference")

    import torch
    from spiral import evaluate, make_batch, make_encoder, objective

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    mode_context = nullcontext()
    if args.device == "tpu":
        if report["metadata"]["torch_tpu_version"] is None:
            raise UnsupportedMode("torch_tpu is not installed in this interpreter")
        import torch_tpu  # noqa: F401 - registers the backend if autoloading is disabled
        if (not hasattr(torch, "tpu") or not torch.accelerator.is_available()
                or torch.accelerator.current_accelerator().type != "tpu"):
            raise UnsupportedMode("TorchTPU reports no available TPU")
        report["metadata"]["device_count"] = torch.accelerator.device_count()
        report["metadata"]["runtime_device_name"] = torch.tpu.get_device_properties().name
        if not args.device_label:
            report["metadata"]["device_label"] = report["metadata"]["runtime_device_name"]
            report["metadata"]["device_label_source"] = "runtime"
        mode = torch.tpu.EagerMode.DEFER_AND_FUSE if args.mode == "fused" else torch.tpu.EagerMode.DEFER_NEVER
        mode_context = torch.tpu.set_eager_mode(mode)
    else:
        report["metadata"]["device_count"] = 1
        if not args.device_label:
            report["metadata"]["device_label"] = platform.processor() or "CPU"

    def sync():
        if args.device == "tpu":
            torch.accelerator.synchronize()

    cfg = report["config"]
    reference = make_encoder(args.width)
    target = deepcopy(reference).to(args.device)
    optimizer = torch.optim.AdamW(target.parameters(), lr=args.lr, weight_decay=args.weight_decay, foreach=False)
    data = torch.Generator(device="cpu").manual_seed(cfg["data_seed"])
    timings = {
        "scope": "Synchronized training wall time: objective, backward, AdamW and Python dispatch. "
                 "Excludes batch generation, transfers, model setup, evaluation, and parity. "
                 "Compile mode wraps the forward objective with fullgraph=True; optimizer runs outside.",
        "first_use": "First training step in this process includes first-use overhead and execution. "
                     "Not a measurement of isolated compiler time.",
        "steady_definition": "Throughput after the first full post-first-step block; residual compilation is possible.",
        "warm_intervals": [], "training_seconds": 0.0, "steady_steps_per_second": None,
    }
    report["timing"] = timings
    with mode_context:
        if args.device == "tpu":
            report["metadata"]["effective_eager_mode"] = str(torch.tpu.eager_mode)
        train_fn = objective
        if args.mode == "compile":
            import torch._dynamo
            torch._dynamo.config.suppress_errors = False
            # Create the wrapper in the first timed section so its first-call overhead is visible.
        finished = 0
        first_inputs = None
        steady_steps, steady_seconds = 0, 0.0
        while finished < args.steps:
            count = 1 if finished == 0 else min(args.block_size, args.steps - finished)
            batches = []
            for _ in range(count):
                host_batch = make_batch(args.batch_size, args.slices, data, rho=args.rho)
                if first_inputs is None:
                    first_inputs = host_batch
                batches.append(tuple(x.to(args.device) for x in host_batch))
            sync()
            started = perf_counter()
            if args.mode == "compile" and finished == 0:
                train_fn = torch.compile(objective, backend="tpu", fullgraph=True, dynamic=False)
            for views, projections in batches:
                optimizer.zero_grad(set_to_none=True)
                loss = train_fn(target, views, projections, args.weight)
                loss.backward()
                optimizer.step()
            sync()
            seconds = perf_counter() - started
            timings["training_seconds"] += seconds
            if finished == 0:
                timings["first_step_seconds"] = seconds
            else:
                timings["warm_intervals"].append({
                    "start_step": finished + 1, "end_step": finished + count, "steps": count,
                    "seconds": seconds, "steps_per_second": count / seconds,
                    "cumulative_seconds": timings["training_seconds"],
                })
                if finished > 1:
                    steady_steps += count
                    steady_seconds += seconds
            finished += count
        timings["steady_steps_per_second"] = steady_steps / steady_seconds if steady_steps else None
        timings["steady_steps"] = steady_steps
        report["final_batch_loss"] = loss.detach().cpu().item()
        report["metrics"], report["scatter"] = evaluate(
            target, args.device, seed=cfg["eval_seed"], n_fit=cfg["probe_fit"],
            n_test=cfg["probe_test"], n_plot=cfg["scatter_samples"],
        )
        # Restore identical initial parameters after measuring so the parity check cannot prewarm the run.
        target.load_state_dict(reference.state_dict())
        target.train()
        report["parity"] = check_parity(torch, reference, target, objective, train_fn, first_inputs, args.device, cfg)
        if not report["parity"]["passed"]:
            raise RuntimeError("One-step loss or gradient differs from the CPU reference beyond the recorded tolerances")


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.device == "tpu":
        os.environ.setdefault("TORCH_TPU_TIER2_COMPILATION_CACHE", "disabled")
    report = setup_report(args)
    try:
        run(args, report)
        report["status"] = "ok"
        json.dumps(report, allow_nan=False)
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "unsupported" if isinstance(exc, (UnsupportedMode, NotImplementedError)) else "error"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    # JSON null preserves the rest of a failed measurement if a diagnostic produced NaN.
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: finite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite(v) for v in value]
        return value

    path = args.output / "result.json"
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps(finite(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)
    print(f"{report['status']}: {path}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
