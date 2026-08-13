#!/usr/bin/env python3
"""Primus GPU MCP server — primus-cli train/benchmark/preflight/projection + Slurm.

Runs ON the Conductor jump host; Cursor connects via SSH stdio.
Submissions default to --dry-run; pass dry_run=false to actually launch.
"""
from __future__ import annotations

import functools
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP

from backend import (
    JUMP,
    LOG_DIR,
    PRIMUS_DIR,
    SOCKET_IFNAME,
    is_local_mode,
    partition_from_nodelist,
    primus_cli,
    ssh_run,
    tail_log_for_job,
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("primus-mcp")

mcp = FastMCP(
    "primus-gpu",
    instructions=(
        "Tools to drive AMD Primus (primus-cli) on the Conductor jump host: "
        "training (Megatron/TorchTitan/MaxText), benchmarks, preflight, projection, "
        "plus Slurm queue control. Training/benchmark submissions default to dry_run=true; "
        "set dry_run=false to actually submit. Prefer sbatch for real training (srun blocks)."
    ),
)


def _tool(fn):
    """Wrap handlers so SSH/Slurm/primus-cli errors return text instead of crashing MCP."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except RuntimeError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # noqa: BLE001 — surface to agent as text
            return f"ERROR: {type(exc).__name__}: {exc}"

    return mcp.tool()(wrapper)


# --------------------------------------------------------------------------- info


@_tool
def primus_info() -> str:
    """Jump host, Primus repo dir, log dir, socket NIC, and execution mode."""
    return json.dumps(
        {
            "jump_host": JUMP,
            "primus_dir": PRIMUS_DIR,
            "log_dir": LOG_DIR,
            "socket_ifname": SOCKET_IFNAME,
            "local_mode": is_local_mode(),
            "note": "submissions default to dry_run=true; use sbatch for real training runs",
        },
        indent=2,
    )


@_tool
def primus_cli_help(topic: str = "") -> str:
    """Show primus-cli help. topic='' for launcher help, or 'train'/'benchmark'/'preflight'/'projection'."""
    t = topic.strip()
    if t and t not in ("train", "benchmark", "preflight", "projection"):
        return "ERROR: topic must be one of: train, benchmark, preflight, projection (or empty)"
    inner = f"-- {t} --help" if t else "--help"
    return primus_cli(inner, mode="direct", dry_run=False, pin_sockets=False, timeout=120)


# --------------------------------------------------------------------------- slurm


@_tool
def slurm_list_jobs() -> str:
    """List the jump user's running/pending Slurm jobs (id name state nodes elapsed)."""
    out = ssh_run("squeue -u ${USER:-sshetkud} -o '%i %j %T %N %M' 2>/dev/null || true")
    return out or "(no jobs in queue)"


@_tool
def slurm_job_status(job_id: str) -> str:
    """squeue + sacct summary for one Slurm job ID."""
    jid = job_id.strip()
    if not jid.isdigit():
        return "ERROR: job_id must be numeric"
    parts = [
        ssh_run(f"squeue -j {jid} -o '%i %j %T %N %M' 2>/dev/null || echo 'not in queue'"),
        ssh_run(
            f"sacct -j {jid} -X -o JobID,JobName,State,ExitCode,Elapsed,Start,End -n 2>/dev/null | head -5 || true"
        ),
    ]
    return "\n---\n".join(parts)


@_tool
def slurm_cancel_job(job_id: str) -> str:
    """Cancel a Slurm job with scancel (verifies it is queued/running first)."""
    jid = job_id.strip()
    if not jid.isdigit():
        return "ERROR: job_id must be numeric"
    before = ssh_run(f"squeue -j {jid} -h -o '%i %j %T %N' 2>/dev/null || true")
    if not before.strip():
        return f"(job {jid} not in queue — nothing to cancel)"
    ssh_run(f"scancel {jid}")
    after = ssh_run(f"squeue -j {jid} -h -o '%i %T' 2>/dev/null || true")
    return f"cancelled: {before}\nafter: {after or '(no longer in queue)'}"


@_tool
def primus_tail_log(job_id: str, lines: int = 60) -> str:
    """Tail the newest primus sbatch log in LOG_DIR matching the job id (read-only)."""
    jid = job_id.strip()
    if not jid.isdigit():
        return "ERROR: job_id must be numeric"
    return tail_log_for_job(jid, lines)


# --------------------------------------------------------------------------- train


@_tool
def primus_submit_train(
    exp_config: str = "examples/maxtext/configs/MI355X/llama3.1_405B-pretrain.yaml",
    nodes: int = 8,
    nodelist: str = "",
    partition: str = "",
    steps: int = 50,
    dataset_type: str = "synthetic",
    max_target_length: int = 8192,
    quantization: str = "fp8",
    per_device_batch_size: int = 5,
    ici_fsdp_parallelism: int = 8,
    dcn_fsdp_parallelism: int = 8,
    ici_tensor_parallelism: int = 1,
    dcn_data_parallelism: int = -1,
    mem_fraction: str = "0.93",
    socket_ifname: str = "enp81s0f1",
    launcher: str = "sbatch",
    suite: str = "pretrain",
    global_config: str = "runner/use_ainic.yaml",
    job_name: str = "primus-train",
    maxtext_path: str = "",
    extra_args: str = "",
    walltime: str = "04:00:00",
    dry_run: bool = True,
    confirm: bool = False,
) -> str:
    """Submit a Primus MaxText/JAX training job, reproducing the validated 405B run (job 7314).

    SAFETY GATE: a real launch requires BOTH dry_run=false AND confirm=true. If dry_run=false
    but confirm is not true, this tool does NOT submit; it returns a CONFIRMATION REQUIRED
    summary of every resolved parameter (and the generated sbatch) and waits. Re-call with
    confirm=true to actually launch. This makes parameter confirmation part of the server.

    ALL run knobs are explicit, typed inputs (no free-form string needed). The agent
    should collect/confirm these with the user BEFORE launching:
      - exp_config: experiment YAML (model). default = Llama-3.1-405B MI355X pretrain.
      - nodes / nodelist: scale + placement (partition auto-derived: r16 -> mi355x-r16).
      - steps: number of training steps.
      - dataset_type: 'synthetic' (throughput) or a real dataset type.
      - max_target_length: sequence length.
      - quantization: precision, e.g. 'fp8' or '' (bf16).
      - per_device_batch_size: micro-batch per GPU (global batch = pdbs * nodes * 8).
      - ici_/dcn_ fsdp/tensor/data parallelism: sharding axes (intra-/inter-node).
      - mem_fraction: XLA_PYTHON_CLIENT_MEM_FRACTION (0.93 validated; '' to omit).
      - socket_ifname: mgmt NIC for NCCL/GLOO control sockets.
      - extra_args: any additional raw MaxText overrides (appended last).

    launcher='sbatch' (recommended) GENERATES + submits a wrapper mirroring the proven
    _primus_405b_srun.sbatch (socket pin, XLA HBM cap, stale-container pre-clean, GPU/IB
    device-settle wait, srun --ntasks-per-node=1 fan-out). launcher='srun' is the direct
    blocking path. dry_run=TRUE by default -> returns the generated sbatch WITHOUT launching.
    """
    exp = exp_config.strip()
    if not exp:
        return "ERROR: exp_config is required"
    if suite not in ("pretrain", "posttrain"):
        return "ERROR: suite must be 'pretrain' or 'posttrain'"
    if launcher not in ("srun", "sbatch"):
        return "ERROR: launcher must be 'srun' or 'sbatch'"

    import shlex as _sh

    ifname = socket_ifname.strip() or SOCKET_IFNAME

    # Compose MaxText overrides from the typed knobs (CLI overrides win over the yaml).
    overrides = [
        "steps=" + str(int(steps)),
        "dataset_type=" + dataset_type.strip(),
        "max_target_length=" + str(int(max_target_length)),
        "per_device_batch_size=" + str(int(per_device_batch_size)),
        "ici_fsdp_parallelism=" + str(int(ici_fsdp_parallelism)),
        "dcn_fsdp_parallelism=" + str(int(dcn_fsdp_parallelism)),
        "ici_tensor_parallelism=" + str(int(ici_tensor_parallelism)),
        "dcn_data_parallelism=" + str(int(dcn_data_parallelism)),
    ]
    if quantization.strip():
        overrides.append("quantization=" + quantization.strip())
    if extra_args.strip():
        overrides += extra_args.split()

    train_args = "train " + suite + " --config " + exp + " " + " ".join(overrides)

    # ---- resolved placement + confirmation summary (shared by both launchers) ----
    _nl = nodelist.strip()
    _part = partition.strip() or (partition_from_nodelist(_nl) if _nl else "")
    _gpus = int(nodes) * 8
    _global_batch = int(per_device_batch_size) * _gpus
    confirm_summary = chr(10).join(
        [
            "CONFIRMATION REQUIRED — review these parameters, then re-call "
            "primus_submit_train with confirm=true (and dry_run=false) to launch.",
            "",
            "Model (exp_config): " + exp,
            "Framework:          MaxText / JAX",
            "Suite:              " + suite,
            "Launcher:           " + launcher,
            "Nodes:              " + str(int(nodes)) + "  (GPUs: " + str(_gpus) + ")",
            "Nodelist:           " + (_nl or "(auto)"),
            "Partition:          " + (_part or "(auto-derived)"),
            "Walltime:           " + walltime,
            "Steps:              " + str(int(steps)),
            "Dataset:            " + dataset_type.strip(),
            "Sequence length:    " + str(int(max_target_length)),
            "Precision:          " + (quantization.strip() or "bf16 (no quantization)"),
            "per_device_batch:   " + str(int(per_device_batch_size)),
            "Global batch:       " + str(int(per_device_batch_size)) + " x " + str(_gpus)
            + " = " + str(_global_batch),
            "ici_fsdp / dcn_fsdp:  " + str(int(ici_fsdp_parallelism)) + " / "
            + str(int(dcn_fsdp_parallelism)),
            "ici_tensor / dcn_data: " + str(int(ici_tensor_parallelism)) + " / "
            + str(int(dcn_data_parallelism)),
            "XLA_PYTHON_CLIENT_MEM_FRACTION: " + (mem_fraction.strip() or "(omitted)"),
            "NCCL/GLOO_SOCKET_IFNAME:        " + ifname,
            "extra_args:         " + (extra_args.strip() or "(none)"),
            "",
            "train args: " + train_args,
        ]
    )
    needs_confirm = (not dry_run) and (not confirm)

    # ---- srun: blocking direct primus-cli path (no wrapper) ----
    if launcher == "srun":
        if needs_confirm:
            return confirm_summary

        return primus_cli(
            train_args,
            mode="slurm",
            launcher="srun",
            nodes=nodes,
            partition=partition,
            nodelist=nodelist.strip(),
            global_config=global_config,
            maxtext_path=maxtext_path,
            dry_run=dry_run,
            timeout=1800,
        )

    # ---- sbatch: generate a wrapper mirroring job 7314 ----
    nnodes = int(nodes)
    nl = nodelist.strip()
    part = partition.strip() or (partition_from_nodelist(nl) if nl else "")
    logf = LOG_DIR + "/" + job_name + "-%j.log"
    gc = ("--config " + global_config + " ") if global_config.strip() else ""

    lines = ["#!/bin/bash", "#SBATCH -N " + str(nnodes)]
    if part:
        lines.append("#SBATCH -p " + part)
    if nl:
        lines.append("#SBATCH --nodelist=" + nl)
    lines += [
        "#SBATCH -J " + job_name,
        "#SBATCH -t " + walltime,
        "#SBATCH -o " + logf,
        "#SBATCH -e " + logf,
        "",
        "export NCCL_SOCKET_IFNAME=" + ifname,
        "export GLOO_SOCKET_IFNAME=" + ifname,
    ]
    if mem_fraction.strip():
        lines.append("export XLA_PYTHON_CLIENT_MEM_FRACTION=" + mem_fraction.strip())
    if maxtext_path.strip():
        lines.append("export MAXTEXT_PATH=" + maxtext_path.strip())
    lines += [
        "cd " + PRIMUS_DIR,
        "",
        "echo '[wrapper] pre-clean stale primus-training container on all nodes'",
        "srun --ntasks-per-node=1 -N " + str(nnodes) + " bash -c 'sudo docker rm -f primus-training >/dev/null 2>&1 || true'",
        "",
        "echo '[wrapper] waiting for GPU/IB devices to settle on all nodes'",
        "srun --ntasks-per-node=1 -N " + str(nnodes) + " bash -c '",
        "  for i in $(seq 1 90); do",
        "    if [ -e /dev/kfd ] && [ -e /dev/dri ] && [ -e /dev/infiniband ]; then",
        '      echo "$(hostname): devices present after $((i*2))s"; exit 0',
        "    fi",
        "    sleep 2",
        "  done",
        '  echo "$(hostname): devices STILL MISSING after 180s" >&2; exit 1',
        "' || { echo '[wrapper] FATAL: GPU/IB devices did not settle on all nodes'; exit 1; }",
        "",
        "echo '[wrapper] launching Primus entry on all nodes via srun'",
        "srun --ntasks-per-node=1 -N " + str(nnodes) + " --kill-on-bad-exit=1 --label bash runner/primus-cli-slurm-entry.sh " + gc + "-- " + train_args,
        "",
    ]
    script = chr(10).join(lines)

    if dry_run:
        return "DRY RUN - generated sbatch wrapper (not submitted):" + chr(10) * 2 + script

    if needs_confirm:
        return (
            confirm_summary
            + chr(10) * 2
            + "--- generated sbatch wrapper (will be submitted on confirm) ---"
            + chr(10) * 2
            + script
        )

    import base64 as _b64

    ssh_run("mkdir -p " + _sh.quote(LOG_DIR) + " 2>/dev/null || true")
    remote_path = LOG_DIR + "/_gen_" + job_name + ".sbatch"
    enc = _b64.b64encode(script.encode()).decode()
    ssh_run("echo " + enc + " | base64 -d > " + _sh.quote(remote_path))
    out = ssh_run("cd " + _sh.quote(PRIMUS_DIR) + " && sbatch " + _sh.quote(remote_path))
    return (
        "submitted via generated wrapper (" + remote_path + "):" + chr(10) + out + chr(10)
        + "(hint: poll with slurm_job_status / primus_tail_log using the returned job id)"
    )

# --------------------------------------------------------------------------- benchmark


@_tool
def primus_benchmark(
    suite: str,
    mode: str = "slurm",
    launcher: str = "srun",
    nodes: int = 1,
    nodelist: str = "",
    gemm_m: int = 4096,
    gemm_n: int = 4096,
    gemm_k: int = 4096,
    extra_args: str = "",
    dry_run: bool = False,
) -> str:
    """Run a Primus microbenchmark.

    suite: gemm | gemm-dense | gemm-deepseek | attention | rccl | strided-allgather
    mode: 'slurm' (multi-node via launcher) or 'direct' (current host/container).
    nodelist: explicit Slurm nodelist for slurm mode (partition auto-derived).
    For gemm-family, --M/--N/--K come from gemm_m/gemm_n/gemm_k; other suites use extra_args.
    """
    valid = {"gemm", "gemm-dense", "gemm-deepseek", "attention", "rccl", "strided-allgather"}
    if suite not in valid:
        return f"ERROR: suite must be one of {sorted(valid)}"
    args = f"benchmark {suite}"
    if suite in ("gemm", "gemm-dense", "gemm-deepseek"):
        args += f" --M {int(gemm_m)} --N {int(gemm_n)} --K {int(gemm_k)}"
    if extra_args.strip():
        args += " " + extra_args.strip()
    return primus_cli(
        args,
        mode=mode,
        launcher=launcher,
        nodes=(nodes if mode == "slurm" else 0),
        nodelist=(nodelist.strip() if mode == "slurm" else ""),
        pin_sockets=(suite == "rccl"),
        dry_run=dry_run,
        timeout=1800,
    )


# --------------------------------------------------------------------------- preflight


@_tool
def primus_preflight(
    nodes: int = 2,
    nodelist: str = "",
    launcher: str = "srun",
    host: bool = True,
    gpu: bool = True,
    network: bool = True,
    perf_test: bool = False,
    report_name: str = "",
    dry_run: bool = False,
) -> str:
    """Run a cluster preflight (host/GPU/network info; optional perf tests) across nodes.

    --host/--gpu/--network are fast info collection; perf_test=true runs GEMM + intra/inter-node comm.
    nodelist: explicit Slurm nodelist (partition auto-derived). report_name -> --report-file-name.
    """
    flags = []
    if host:
        flags.append("--host")
    if gpu:
        flags.append("--gpu")
    if network:
        flags.append("--network")
    if perf_test:
        flags.append("--perf-test")
    if report_name.strip():
        import shlex as _sh

        flags.append(f"--report-file-name {_sh.quote(report_name.strip())}")
    args = "preflight " + " ".join(flags) if flags else "preflight"
    return primus_cli(
        args, mode="slurm", launcher=launcher, nodes=nodes, nodelist=nodelist.strip(),
        dry_run=dry_run, timeout=1800,
    )


# --------------------------------------------------------------------------- projection


@_tool
def primus_projection(suite: str = "performance", extra_args: str = "", dry_run: bool = False) -> str:
    """Run Primus performance/memory projection (no GPUs needed; runs in direct mode).

    suite: 'memory' (per-GPU memory analysis) or 'performance' (throughput + multinode scaling).
    Pass overrides via extra_args, e.g. '--target-nodes 16 --micro-batch-size 1 --deepep'.
    """
    if suite not in ("memory", "performance"):
        return "ERROR: suite must be 'memory' or 'performance'"
    args = f"projection {suite}"
    if extra_args.strip():
        args += " " + extra_args.strip()
    return primus_cli(args, mode="direct", pin_sockets=False, dry_run=dry_run, timeout=300)


# --------------------------------------------------------------------------- gpu util


def _resolve_job_nodes(jid: str) -> list[str]:
    """Expand a running/finished job's allocation into a list of short hostnames."""
    import shlex as _sh

    nl = ssh_run(
        f"scontrol show job {jid} -o 2>/dev/null | tr ' ' '\\n' | grep -m1 '^NodeList=' | cut -d= -f2"
    ).strip()
    if not nl or nl in ("None", "(null)", "None assigned"):
        nl = ssh_run(f"squeue -j {jid} -h -O 'NodeList:5000' 2>/dev/null | head -1").strip()
    if not nl:
        nl = ssh_run(f"sacct -j {jid} -X -n -o 'NodeList%-5000' 2>/dev/null | head -1").strip()
    if not nl or nl in ("None", "(null)", "None assigned"):
        return []
    hosts = ssh_run(
        f"scontrol show hostnames {_sh.quote(nl)} 2>/dev/null || echo {_sh.quote(nl)}"
    )
    return [h.strip() for h in hosts.split() if h.strip()]


def _num(v):
    """Coerce an amd-smi field ({'value': N, 'unit': ...}, number, or string) to float."""
    if isinstance(v, dict):
        v = v.get("value")
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.split()[0])
        except (ValueError, IndexError):
            return None
    return None


def _find_val(obj, keys):
    """Depth-first search for the first matching key anywhere in a nested amd-smi dict."""
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k in keys:
                    n = _num(v)
                    if n is not None:
                        return n
                stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


def _amd_smi_gpus(fqdn: str) -> list:
    """SSH to a node, run amd-smi, and return its parsed JSON list of GPU objects."""
    import shlex as _sh

    inner = (
        "amd-smi metric --usage --mem-usage --json 2>/dev/null "
        "|| /opt/rocm/bin/amd-smi metric --usage --mem-usage --json 2>/dev/null"
    )
    remote = (
        f"ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 "
        f"{_sh.quote(fqdn)} {_sh.quote(inner)}"
    )
    raw = ssh_run(remote, timeout=90)
    if not raw or not raw.strip():
        raise RuntimeError(f"amd-smi returned nothing on {fqdn} (not installed / no GPUs?)")
    start = min([p for p in (raw.find("["), raw.find("{")) if p != -1] or [-1])
    if start < 0:
        raise RuntimeError(f"amd-smi output not JSON on {fqdn}: {raw[:200]}")
    data = json.loads(raw[start:])
    if isinstance(data, dict):
        for key in ("gpu_data", "gpus", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            lists = [v for v in data.values() if isinstance(v, list)]
            data = lists[0] if lists else [data]
    return data if isinstance(data, list) else [data]


@_tool
def gpu_utilization(job_id: str = "", node: str = "", per_gpu: bool = True) -> str:
    """Snapshot GPU compute + VRAM utilization via amd-smi on a job's node(s).

    Resolves the target from either an explicit `node` (full name or shorthand,
    e.g. smci355-ccs-aus-r17-06 or r17-06) or a Slurm `job_id` (multi-node
    allocations are expanded with scontrol). SSHes to each node and runs
    `amd-smi metric --usage --mem-usage --json`, reporting per-GPU gfx activity
    (%), VRAM used/total (GiB), and VRAM used (%). Ideal for watching a running
    Primus training job's GPUs. Read-only.
    - job_id: running/recent Slurm job (used when node is empty).
    - node: explicit target node; overrides job_id.
    - per_gpu: True for one row per GPU; False for a per-node summary
               (mean gfx%, summed VRAM).
    """
    node = node.strip()
    jid = job_id.strip()
    if node:
        nodes = [node.split(".")[0]]
    elif jid:
        if not jid.isdigit():
            return "ERROR: job_id must be numeric"
        nodes = _resolve_job_nodes(jid)
        if not nodes:
            return f"ERROR: could not resolve node(s) for job {jid} (not in squeue/sacct)"
    else:
        return "ERROR: provide either node or job_id"

    gfx_keys = {"gfx_activity", "gfx_activity_percent", "graphics_activity", "gfx"}
    used_keys = {"used_vram", "vram_used", "used", "vram_used_mb"}
    total_keys = {"total_vram", "vram_total", "total", "vram_total_mb"}

    rows = []
    errors = []
    for short in nodes:
        fqdn = short if "." in short else f"{short}.cs-aus.dcgpu"
        try:
            gpus = _amd_smi_gpus(fqdn)
        except (RuntimeError, ValueError) as exc:
            errors.append(f"{short}: {exc}")
            continue
        for idx, g in enumerate(gpus):
            gid = _find_val(g, {"gpu", "gpu_id", "card"})
            gfx = _find_val(g, gfx_keys)
            used = _find_val(g, used_keys)
            total = _find_val(g, total_keys)
            used_gib = used / 1024.0 if used is not None else None
            total_gib = total / 1024.0 if total is not None else None
            vram_pct = (100.0 * used / total) if (used and total) else None
            rows.append(
                {
                    "node": short,
                    "gpu": int(gid) if gid is not None else idx,
                    "gfx": gfx,
                    "used": used_gib,
                    "total": total_gib,
                    "vram_pct": vram_pct,
                }
            )

    if not rows:
        return "ERROR: no GPU data collected\n" + "\n".join(errors)

    def fmt(v, p=1):
        return f"{v:.{p}f}" if isinstance(v, (int, float)) else "-"

    lines = []
    if per_gpu:
        lines.append(f"{'Node':<24} {'GPU':>3} {'GFX%':>6} {'VRAM(GiB)':>18} {'VRAM%':>6}")
        for r in sorted(rows, key=lambda r: (r["node"], r["gpu"])):
            vram = f"{fmt(r['used'])}/{fmt(r['total'])}"
            lines.append(
                f"{r['node']:<24} {r['gpu']:>3} {fmt(r['gfx']):>6} {vram:>18} {fmt(r['vram_pct']):>6}"
            )
    else:
        lines.append(f"{'Node':<24} {'GPUs':>4} {'meanGFX%':>9} {'VRAM used/total(GiB)':>24}")
        by_node: dict[str, list] = {}
        for r in rows:
            by_node.setdefault(r["node"], []).append(r)
        for short in sorted(by_node):
            g = by_node[short]
            gfxs = [x["gfx"] for x in g if isinstance(x["gfx"], (int, float))]
            used = sum(x["used"] for x in g if isinstance(x["used"], (int, float)))
            total = sum(x["total"] for x in g if isinstance(x["total"], (int, float)))
            mean_gfx = sum(gfxs) / len(gfxs) if gfxs else None
            lines.append(
                f"{short:<24} {len(g):>4} {fmt(mean_gfx):>9} {fmt(used) + '/' + fmt(total):>24}"
            )
    if errors:
        lines.append("")
        lines.append("errors:")
        lines.extend(f"  {e}" for e in errors)
    return "\n".join(lines)


if __name__ == "__main__":
    log.info("primus-gpu MCP starting jump=%s primus_dir=%s local_mode=%s", JUMP, PRIMUS_DIR, is_local_mode())
    mcp.run()
