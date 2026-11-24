#!/usr/bin/env python3
"""
bench.py - build, run, verify and benchmark every implementation in one algorithm folder.

Run it from an algorithm folder (it benchmarks the .c / .cpp / .cu files sitting there):

    cd 01_reduction/sum
    python3 ../../common/bench.py                                  # defaults
    python3 ../../common/bench.py --sizes 1e7,1e8 --reps 7 --threads 1,2,4 --plot

Or copy this file into each folder and run `python3 bench.py`.

------------------------------------------------------------------------------
CONTRACT every program must follow
------------------------------------------------------------------------------
  usage:   ./prog <n>                      (thread count comes from OMP_NUM_THREADS)
  output:  one line on stdout, the last line starting with RESULT:

      RESULT time_ms=12.345 result=50004842.960109
      RESULT time_ms=1.2 h2d_ms=40.1 d2h_ms=0.02 result=123456,7.5     (CUDA)

    time_ms  compute/kernel time only (no data generation, no malloc, no transfers)
    h2d_ms / d2h_ms   optional; host<->device copies (CUDA)
    result   what the algorithm computed, compared against the reference. Several
             values are comma-separated, no spaces (e.g. "min,argmin"). Integer tokens
             must match exactly, float tokens match within --rtol. For sorting, print a
             checksum / sorted-flag instead of the array.

  Every program must generate the SAME input for a given n (fixed seed).
  float32 programs: accumulate the reference in double (or Kahan). A naive float running
  sum of ~1e8 values in [0,1) stalls near 2^24 and is simply wrong, so no rtol fixes it.
  CUDA programs must warm up inside the process (cudaFree(0); one untimed launch),
  because context creation and first-launch cost live inside the process.

------------------------------------------------------------------------------
Per-file tags (put in the first 20 lines of a source file, inside a comment)
------------------------------------------------------------------------------
  bench-flags: -ffast-math -march=native      extra compiler flags for this file only
  bench-role: reference                       this file is the reference (default: seq.c)
  bench-skip                                  ignore this file

Per-algorithm defaults can live in bench.cfg (JSON) next to the sources, e.g.
  {"bytes_per_elem": 8, "flops_per_elem": 1, "sizes": "1e7,1e8", "rtol": 1e-9}
"""
import argparse
import csv
import json
import math
import os
import platform
import re
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Peak DRAM bandwidth (GB/s) for GPUs we know; override with --peak-bw-gpu
KNOWN_GPU_PEAK_BW = {"T4": 320.0, "L4": 300.0}

SRC_COMPILER = {".c": "gcc", ".cpp": "g++", ".cu": "nvcc"}


# ----------------------------------------------------------------------------
# environment detection
# ----------------------------------------------------------------------------
def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def gpu_info():
    """Return (name, sm_arch) or (None, None) if there is no usable NVIDIA GPU."""
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True).stdout
        name, cc = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return name, "sm_" + cc.replace(".", "")
    except Exception:
        pass
    try:  # older drivers do not know compute_cap
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True).stdout
        return out.strip().splitlines()[0].strip(), None
    except Exception:
        return None, None


def tool_version(cmd):
    if not shutil.which(cmd[0]):
        return "n/a"
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
        return out.strip().splitlines()[-1 if cmd[0] == "nvcc" else 0].strip()
    except Exception:
        return "n/a"


# ----------------------------------------------------------------------------
# source discovery
# ----------------------------------------------------------------------------
@dataclass
class Impl:
    name: str
    path: Path
    kind: str                      # seq | omp | cuda
    flags: list = field(default_factory=list)
    role: str = ""
    binary: Path = None
    build_error: str = ""


def read_tags(path):
    flags, role, skip = [], "", False
    with open(path, errors="ignore") as f:
        head = [f.readline() for _ in range(20)]
    for line in head:
        m = re.search(r"bench-flags:\s*(.*)", line)
        if m:
            flags = shlex.split(m.group(1).split("*/")[0].strip())
        m = re.search(r"bench-role:\s*(\w+)", line)
        if m:
            role = m.group(1)
        if "bench-skip" in line:
            skip = True
    return flags, role, skip


def discover(folder, only):
    impls = []
    for p in sorted(folder.iterdir()):
        if p.suffix not in SRC_COMPILER or p.name.startswith("_"):
            continue
        if only and p.stem not in only:
            continue
        flags, role, skip = read_tags(p)
        if skip:
            continue
        if p.suffix == ".cu":
            kind = "cuda"
        else:
            text = p.read_text(errors="ignore")
            kind = "omp" if re.search(r"#\s*pragma\s+omp|omp\.h", text) else "seq"
        impls.append(Impl(p.stem, p, kind, flags, role))
    return impls


def pick_reference(impls, wanted):
    if wanted:
        for i in impls:
            if i.name == wanted:
                return i
        sys.exit(f"error: --reference {wanted!r} not found among {[i.name for i in impls]}")
    for i in impls:
        if i.role == "reference":
            return i
    for i in impls:
        if i.name == "seq":
            return i
    for i in impls:
        if i.kind == "seq":
            return i
    sys.exit("error: no reference implementation (add seq.c or tag a file with 'bench-role: reference')")


# ----------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------
def build(impl, bin_dir, arch):
    compiler = SRC_COMPILER[impl.path.suffix]
    if not shutil.which(compiler):
        impl.build_error = f"SKIP: {compiler} not found"
        return
    out = bin_dir / impl.name
    if impl.kind == "cuda":
        cmd = ["nvcc", "-O3", "-std=c++17", "-lineinfo"] + ([f"-arch={arch}"] if arch else [])
    else:
        std = "-std=c++17" if impl.path.suffix == ".cpp" else "-std=c11"
        cmd = [compiler, "-O3", std, "-Wall", "-Wextra"]
        if impl.kind == "omp":
            cmd.append("-fopenmp")
    cmd += impl.flags + ["-o", str(out), str(impl.path)]
    if impl.kind != "cuda":
        cmd.append("-lm")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        impl.build_error = (p.stderr or p.stdout).strip()
        return
    if p.stderr.strip():  # keep compiler warnings visible
        print(f"  warnings in {impl.path.name}:\n    " + p.stderr.strip().replace("\n", "\n    "))
    impl.binary = out


# ----------------------------------------------------------------------------
# run + verify
# ----------------------------------------------------------------------------
def run_once(impl, n, threads, args):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads if threads else 1)
    if threads and not args.no_pin:
        env.setdefault("OMP_PROC_BIND", "close")
        env.setdefault("OMP_PLACES", "cores")
    try:
        p = subprocess.run([str(impl.binary), str(n)], capture_output=True,
                           text=True, timeout=args.timeout, env=env)
    except subprocess.TimeoutExpired:
        return None, f"timeout after {args.timeout}s"
    if p.returncode != 0:
        return None, f"exit code {p.returncode}: {p.stderr.strip()[:200]}"
    line = None
    for l in p.stdout.splitlines():
        if l.startswith("RESULT"):
            line = l
    if line is None:
        return None, "no RESULT line in output"
    fields = dict(tok.split("=", 1) for tok in line.split()[1:] if "=" in tok)
    if "time_ms" not in fields or "result" not in fields:
        return None, "RESULT line needs time_ms= and result="
    try:
        fields["time_ms"] = float(fields["time_ms"])
        for k in ("h2d_ms", "d2h_ms"):
            if k in fields:
                fields[k] = float(fields[k])
    except ValueError:
        return None, f"bad number in: {line}"
    return fields, None


def measure(impl, n, threads, args):
    """warmup runs (discarded) followed by timed runs; returns (runs, error)"""
    for _ in range(args.warmup):
        _, err = run_once(impl, n, threads, args)
        if err:
            return None, err
    runs = []
    for _ in range(args.reps):
        f, err = run_once(impl, n, threads, args)
        if err:
            return None, err
        runs.append(f)
    return runs, None


def compare(got, ref, rtol, atol):
    """Compare two comma-separated result strings. Returns (ok, worst_relative_error)."""
    g, r = got.split(","), ref.split(",")
    if len(g) != len(r):
        return False, math.inf
    worst = 0.0
    for a, b in zip(g, r):
        try:
            if int(a) != int(b):
                return False, math.inf
            continue
        except ValueError:
            pass
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            if a != b:
                return False, math.inf
            continue
        if math.isnan(fa) or math.isnan(fb):
            return False, math.inf
        diff = abs(fa - fb)
        worst = max(worst, diff / max(abs(fb), 1e-300))
        if diff > max(rtol * max(abs(fa), abs(fb)), atol):
            return False, worst
    return True, worst


def summarize(runs):
    t = [r["time_ms"] for r in runs]
    s = {"time_ms": statistics.median(t), "time_min": min(t),
         "cv_pct": (statistics.stdev(t) / statistics.mean(t) * 100) if len(t) > 1 else 0.0}
    for k in ("h2d_ms", "d2h_ms"):
        if all(k in r for r in runs):
            s[k] = statistics.median(r[k] for r in runs)
    if "h2d_ms" in s or "d2h_ms" in s:
        s["total_ms"] = s["time_ms"] + s.get("h2d_ms", 0.0) + s.get("d2h_ms", 0.0)
    return s


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------
COLS = [("impl", 24, "<"), ("thr", 4, ">"), ("med_ms", 10, ">"), ("min_ms", 10, ">"),
        ("cv%", 5, ">"), ("GB/s", 8, ">"), ("GFLOPS", 8, ">"), ("%peakBW", 8, ">"),
        ("speedup", 8, ">"), ("spd+xfer", 8, ">"), ("effic", 6, ">"), ("verify", 12, "<")]


def fmt(v, spec=".3f"):
    return "-" if v is None else format(v, spec)


def print_header(n):
    print(f"\nn = {n:,}")
    print("  " + " ".join(f"{c:{a}{w}}" for c, w, a in COLS))
    print("  " + " ".join("-" * w for _, w, _ in COLS))


def print_row(r):
    vals = [r["impl"], str(r["threads"] or "-"),
            fmt(r["time_ms"], ".3f"), fmt(r["time_min"], ".3f"), fmt(r["cv_pct"], ".1f"),
            fmt(r["gbps"], ".1f"), fmt(r["gflops"], ".2f"), fmt(r["pct_peak_bw"], ".1f"),
            fmt(r["speedup"], ".2f") + ("x" if r["speedup"] is not None else ""),
            fmt(r["speedup_total"], ".2f") + ("x" if r["speedup_total"] is not None else ""),
            fmt(r["efficiency"], ".2f"), r["status"]]
    print("  " + " ".join(f"{v:{a}{w}}" for v, (_, w, a) in zip(vals, COLS)))


def make_plot(rows, path, algo):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed, skipping plot)")
        return
    good = [r for r in rows if r["time_ms"] is not None]
    if not good:
        return
    impls = sorted({r["impl"] for r in good})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for name in impls:
        rs = [r for r in good if r["impl"] == name]
        maxthr = max((r["threads"] or 0) for r in rs)
        rs = sorted([r for r in rs if (r["threads"] or 0) == maxthr], key=lambda r: r["n"])
        label = f"{name} ({maxthr} thr)" if maxthr else name
        ax[0].plot([r["n"] for r in rs], [r["time_ms"] for r in rs], marker="o", label=label)
        ax[1].plot([r["n"] for r in rs], [r["gbps"] for r in rs], marker="o", label=label)
    ax[0].set(xscale="log", yscale="log", xlabel="n", ylabel="median time (ms)", title="Time vs n")
    ax[1].set(xscale="log", xlabel="n", ylabel="effective GB/s", title="Effective bandwidth")
    ax[0].legend(fontsize=7)
    nmax = max(r["n"] for r in good)
    omp = [r for r in good if r["kind"] == "omp" and r["n"] == nmax and r["speedup"] is not None]
    for name in sorted({r["impl"] for r in omp}):
        rs = sorted([r for r in omp if r["impl"] == name], key=lambda r: r["threads"])
        ax[2].plot([r["threads"] for r in rs], [r["speedup"] for r in rs], marker="o", label=name)
    if omp:
        tmax = max(r["threads"] for r in omp)
        ax[2].plot([1, tmax], [1, tmax], "k--", linewidth=0.8, label="ideal")
        ax[2].legend(fontsize=7)
    ax[2].set(xlabel="threads", ylabel="speedup vs baseline", title=f"OpenMP scaling (n={nmax:,})")
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.suptitle(algo)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"plot written to {path}")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dir", default=".")
    known, _ = pre.parse_known_args()
    cfg = {}
    cfg_path = Path(known.dir) / "bench.cfg"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())

    ap = argparse.ArgumentParser(description="Build, run, verify and benchmark every .c/.cpp/.cu in a folder.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dir", default=".", help="folder with the sources")
    ap.add_argument("--sizes", default="1e6,1e7,1e8", help="comma-separated element counts")
    ap.add_argument("--threads", default=None, help="OpenMP thread counts (default: powers of 2 up to #cores)")
    ap.add_argument("--reps", type=int, default=5, help="timed runs per configuration")
    ap.add_argument("--warmup", type=int, default=1, help="discarded runs before timing")
    ap.add_argument("--timeout", type=int, default=300, help="seconds per run")
    ap.add_argument("--only", default=None, help="comma-separated file stems to run")
    ap.add_argument("--reference", default=None, help="reference impl (default: seq or bench-role tag)")
    ap.add_argument("--baseline", default=None, help="impl speedups are measured against (default: reference)")
    ap.add_argument("--bytes-per-elem", type=float, default=8.0, help="bytes moved per element (sum: 8)")
    ap.add_argument("--flops-per-elem", type=float, default=1.0, help="flops per element (sum: 1)")
    ap.add_argument("--rtol", type=float, default=1e-9, help="relative tolerance for float results (double: 1e-9; float32: ~1e-4, with a double-accumulating reference)")
    ap.add_argument("--atol", type=float, default=0.0, help="absolute tolerance for float results")
    ap.add_argument("--peak-bw-gpu", type=float, default=None, help="GPU peak GB/s (auto for T4/L4)")
    ap.add_argument("--peak-bw-cpu", type=float, default=None, help="CPU DRAM peak GB/s")
    ap.add_argument("--arch", default=None, help="nvcc arch, e.g. sm_75 (default: auto-detect)")
    ap.add_argument("--no-pin", action="store_true", help="do not set OMP_PROC_BIND/OMP_PLACES")
    ap.add_argument("--no-build", action="store_true", help="reuse existing binaries in bin/")
    ap.add_argument("--plot", action="store_true", help="write results/<algo>.png (needs matplotlib)")
    ap.add_argument("--tag", default="", help="write results to results/<tag>/ (e.g. colab_T4)")
    ap.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items()})
    return ap.parse_args()


def as_list(value):
    """Accept '1e6,1e7' (CLI) or [1000000, 1e7] (JSON in bench.cfg)."""
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    return [x for x in items if str(x).strip()]


def main():
    args = parse_args()
    folder = Path(args.dir).resolve()
    algo = folder.name
    bin_dir = folder / "bin"
    res_dir = folder / "results" / args.tag if args.tag else folder / "results"
    bin_dir.mkdir(exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    sizes = [int(float(s)) for s in as_list(args.sizes)]
    ncpu = os.cpu_count() or 1
    if args.threads:
        thread_list = [int(t) for t in as_list(args.threads)]
    else:
        thread_list = sorted({t for t in (1, 2, 4, 8, 16, 32, 64) if t <= ncpu} | {ncpu})

    impls = discover(folder, {str(x) for x in as_list(args.only)} if args.only else None)
    if not impls:
        sys.exit(f"no .c/.cpp/.cu files found in {folder}")
    ref = pick_reference(impls, args.reference)
    baseline_name = args.baseline or ref.name

    gpu, arch = gpu_info()
    arch = args.arch or arch
    cpu = cpu_name()
    peak_gpu = args.peak_bw_gpu or next((v for k, v in KNOWN_GPU_PEAK_BW.items() if gpu and k in gpu), None)

    print(f"algorithm : {algo}")
    print(f"cpu       : {cpu}  ({ncpu} logical cores)")
    print(f"gpu       : {gpu or 'none detected'}" + (f"  [{arch}]" if arch else ""))
    print(f"compilers : {tool_version(['gcc', '--version'])} | {tool_version(['nvcc', '--version'])}")
    print(f"reference : {ref.name}   baseline for speedup: {baseline_name}")
    print(f"sizes     : {sizes}   omp threads: {thread_list}   warmup={args.warmup} reps={args.reps}")

    # ---- build ----------------------------------------------------------
    if args.no_build:
        for i in impls:
            i.binary = bin_dir / i.name
            if not i.binary.exists():
                i.build_error = "binary missing (run without --no-build)"
    else:
        print("\nbuilding:")
        for i in impls:
            build(i, bin_dir, arch)
            state = "ok" if not i.build_error else ("skipped" if i.build_error.startswith("SKIP") else "FAILED")
            print(f"  {i.path.name:32s} [{i.kind:4s}] {state}")
            if i.build_error:
                print("    " + i.build_error.replace("\n", "\n    "))
    if ref.build_error:
        sys.exit("error: the reference implementation did not build")

    # ---- run ------------------------------------------------------------
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rows, any_fail = [], False
    order = [ref] + [i for i in impls if i is not ref]

    for n in sizes:
        print_header(n)
        n_rows, ref_result = [], None
        for impl in order:
            configs = thread_list if impl.kind == "omp" else [None]
            for thr in configs:
                row = dict(algo=algo, impl=impl.name, kind=impl.kind, n=n, threads=thr,
                           reps=args.reps, time_ms=None, time_min=None, cv_pct=None,
                           h2d_ms=None, d2h_ms=None, total_ms=None, gbps=None, gflops=None,
                           pct_peak_bw=None, speedup=None, speedup_total=None, efficiency=None,
                           max_rel_err=None, status="", cpu=cpu, gpu=gpu or "", timestamp=stamp)
                if impl.build_error:
                    row["status"] = "SKIPPED" if impl.build_error.startswith("SKIP") else "BUILD FAIL"
                elif impl.kind == "cuda" and not gpu:
                    row["status"] = "no GPU"
                else:
                    print(f"    running {impl.name} n={n:,}" + (f" threads={thr}" if thr else "") + " ...",
                          end="\r", file=sys.stderr, flush=True)
                    runs, err = measure(impl, n, thr, args)
                    print(" " * 70, end="\r", file=sys.stderr)
                    if err:
                        row["status"] = "RUN FAIL"
                        print(f"  [{impl.name} thr={thr} n={n:,}] {err}")
                    else:
                        s = summarize(runs)
                        if impl is ref and ref_result is None:
                            ref_result = runs[0]["result"]
                        # every timed run must agree with the reference (catches races)
                        oks, errs = zip(*(compare(r["result"], ref_result, args.rtol, args.atol)
                                          for r in runs)) if ref_result is not None else ((False,), (math.inf,))
                        row["status"] = "PASS" if all(oks) else (
                            "FAIL (varies)" if any(oks) else "FAIL (wrong)")
                        row["max_rel_err"] = max(errs)
                        t_s = s["time_ms"] / 1e3
                        row.update(time_ms=s["time_ms"], time_min=s["time_min"], cv_pct=s["cv_pct"],
                                   h2d_ms=s.get("h2d_ms"), d2h_ms=s.get("d2h_ms"), total_ms=s.get("total_ms"),
                                   gbps=n * args.bytes_per_elem / t_s / 1e9,
                                   gflops=n * args.flops_per_elem / t_s / 1e9)
                        peak = peak_gpu if impl.kind == "cuda" else args.peak_bw_cpu
                        if peak:
                            row["pct_peak_bw"] = row["gbps"] / peak * 100
                if row["status"] not in ("PASS", "no GPU", "SKIPPED"):
                    any_fail = True
                n_rows.append(row)

        # speedups need the baseline row for this n
        base = next((r for r in n_rows if r["impl"] == baseline_name and r["time_ms"]), None)
        for r in n_rows:
            if base and r["time_ms"]:
                r["speedup"] = base["time_ms"] / r["time_ms"]
                if r["total_ms"]:
                    r["speedup_total"] = base["time_ms"] / r["total_ms"]
                if r["kind"] == "omp":
                    r["efficiency"] = r["speedup"] / r["threads"]
            print_row(r)
        rows.extend(n_rows)

    # ---- write results --------------------------------------------------
    fields = ["timestamp", "algo", "impl", "kind", "n", "threads", "reps", "time_ms", "time_min",
              "cv_pct", "h2d_ms", "d2h_ms", "total_ms", "gbps", "gflops", "pct_peak_bw", "speedup",
              "speedup_total", "efficiency", "max_rel_err", "status", "cpu", "gpu"]
    out = res_dir / f"{algo}_{stamp}.csv"
    for path in (out, res_dir / "latest.csv"):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows({k: ("" if r[k] is None else r[k]) for k in fields} for r in rows)
    print(f"\nresults written to {out}  (and results/latest.csv)")
    if args.plot:
        make_plot(rows, res_dir / f"{algo}.png", algo)

    if any_fail:
        print("\nSOME CONFIGURATIONS FAILED - check the verify column above.")
        sys.exit(1)
    print("\nall configurations built, ran and verified.")


if __name__ == "__main__":
    main()