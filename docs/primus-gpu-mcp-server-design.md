# primus-gpu MCP Server — Design & Architecture

How the **`primus-gpu`** Model Context Protocol (MCP) server is designed, deployed, and wired so a
Cursor agent can drive AMD **Primus** (`primus-cli`) training / benchmark / preflight / projection
jobs on AMD MI355X (`mi355x-r16`, `mi355x-r17`) from the Conductor jump host — submit MaxText /
Megatron / TorchTitan runs, inspect them, and cancel — reproducing the validated Llama-3.1-405B
launch (job 7314) on every submission.

> **Key framing:** The MCP server process runs on the **Conductor jump host** (`smc200x`), not on
> the laptop and not on the GPU node. Cursor connects to it over **SSH stdio**. Because the server
> lives on (or SSHes into) the jump host, its tools call `squeue` / `sbatch` / `scancel` / `srun`
> and `./primus-cli` as plain subprocesses — no nested SSH, no per-call re-auth. The correctness
> fixes it relies on live **on disk in the Primus tree**, not in the MCP; the server is a thin,
> reproducible **launcher**.

---

## 1. What runs where

| Location | Component | Container? | Port |
|----------|-----------|------------|------|
| **Laptop** | Cursor (MCP client) | No | — |
| **Jump host** (`smc200x`) | **`server.py`** (FastMCP server, in `.venv`) | No — host process | stdio (over SSH) |
| **Jump host** | `backend.py` command helpers (`ssh_run`, `primus_cli`) | No | — |
| **Jump host** | `logs/` (sbatch stdout/err + generated `_gen_<job>.sbatch`) | No | — |
| **GPU nodes** (8 × `r16`/`r17`) | Slurm job → Docker (`primus-training`, `rocm/primus`) | Yes | — |
| **Shared NFS** (`/mnt/dcgpuval/...`) | Primus tree, configs, logs, model weights | No | — |

NFS is shared, so job logs and the Primus repo are read/written directly on the jump host without
SSH to the compute nodes.

---

## 2. Transport & topology

```
Cursor (laptop)
   │  MCP stdio (JSON-RPC 2.0) tunneled over SSH
   ▼
server.py  (jump host smc200x, .venv/bin/python)
   │  local subprocess (or ssh): ./primus-cli · sbatch · squeue · scancel · tail
   │  sbatch <generated wrapper>  ──►  srun --ntasks-per-node=1 -N 8  (fan-out to all nodes)
   ▼
Slurm controller · 8× MI355X compute nodes · rocm/primus containers · NFS logs · weights
```

Cursor spawns **one** `server.py` process and keeps the stdio pipe open for the whole session.

---

## 3. Two-layer design: transport vs. cluster I/O

The server is deliberately split so tools stay tiny and portable.

| Layer | File | Responsibility |
|-------|------|----------------|
| **Tool surface** | `server.py` | Declares MCP tools, validates args, composes MaxText overrides, generates the wrapper sbatch, formats results. No knowledge of *how* commands run. |
| **Cluster I/O** | `backend.py` | Runs commands locally or over SSH, builds `primus-cli` invocations, derives partitions, tails NFS logs. Auto-detects local vs. remote mode. |

`backend.py` auto-detects whether it is already on the jump host (subprocess) or on a laptop
(ssh first), so the identical tool code works in both deployments:

```python
def _on_jump_host() -> bool:
    host = socket.gethostname().lower()
    return "smc200x" in host or host.endswith(".cs-aus.dcgpu")
```

Core helpers:

| Helper | Purpose |
|--------|---------|
| `ssh_run(cmd, timeout=)` | Run a shell command locally or over SSH; capture stdout+stderr; raise `RuntimeError` on non-zero exit. |
| `primus_cli(args, mode, launcher, …)` | Assemble `cd $PRIMUS_DIR && ./primus-cli [--dry-run] [--config gc] <mode> [launcher flags] -- <args>`. |
| `partition_from_nodelist(nodelist)` | Derive the Slurm partition from the rack (`r16 -> mi355x-r16`). |
| `tail_log_for_job(job_id, lines)` | Tail the newest `logs/*<jobid>*.log`. |

Config is environment-driven (no hardcoded paths):

| Var | Default |
|-----|---------|
| `PRIMUS_JUMP_HOST` | `sshetkud@smc200x-ccs-e12-31.cs-aus.dcgpu` |
| `PRIMUS_DIR` | `/mnt/dcgpuval/afde/sshetkud/Primus` (the fixed tree) |
| `PRIMUS_LOG_DIR` | `/mnt/dcgpuval/afde/sshetkud/primus-mcp/logs` |
| `PRIMUS_SOCKET_IFNAME` | `enp81s0f1` (mgmt NIC for NCCL/GLOO control sockets) |
| `PRIMUS_MCP_LOCAL` | auto (forces local/subprocess mode on the jump host) |

---

## 4. Tool pattern

Every tool is a plain, type-hinted Python function; FastMCP turns the signature into the JSON
schema the agent sees. A shared decorator enforces three conventions.

```python
def _tool(fn):
    """Wrap tool handlers so SSH/Slurm/primus-cli errors return text instead of crashing MCP."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except RuntimeError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"
    return mcp.tool()(wrapper)
```

1. **Return text, never raise.** Any failure becomes a readable string the agent can reason about.
2. **Validate + sanitize inputs.** Job IDs must be numeric (`jid.isdigit()`); every value
 interpolated into a shell string is `shlex.quote(...)`-escaped.
3. **Defensive shell.** Commands use `|| true` so "nothing found" is a normal result, not an error.

---

## 5. Tool catalog

### Read / status

| Tool | Args | Action |
|------|------|--------|
| `primus_info` | — | Jump host, Primus dir, log dir, socket NIC, execution mode |
| `primus_cli_help` | `topic` | `primus-cli` help passthrough (`train`/`benchmark`/`preflight`/`projection`) |
| `slurm_list_jobs` | — | List the user's running/pending jobs |
| `slurm_job_status` | `job_id` | `squeue` + `sacct` summary for one job |
| `primus_tail_log` | `job_id, lines` | Tail the newest sbatch log for the job (read-only) |

### Act (write)

| Tool | Args (key) | Action |
|------|------------|--------|
| `primus_submit_train` | `exp_config, nodes, nodelist, steps, dataset_type, max_target_length, quantization, per_device_batch_size, ici_/dcn_ {fsdp,tensor,data}, mem_fraction, socket_ifname, launcher, dry_run` | Compose MaxText overrides → generate + submit the 7314 wrapper sbatch |
| `primus_benchmark` | `suite, mode, …` | Run a primus-cli benchmark suite |
| `primus_preflight` | `…` | Pre-run environment / config checks |
| `primus_projection` | `suite, extra_args` | Perf / memory projection |
| `slurm_cancel_job` | `job_id` | Verify in queue, then `scancel` |
| `gpu_utilization` | `job_id, node, per_gpu` | `amd-smi` snapshot on the job's / node's GPUs |

---

## 6. The `primus_submit_train` flow (the validated path)

All run knobs are **explicit, typed inputs** (defaults = the validated 405B values). The agent
confirms them with the user before launching. The tool then, for `launcher='sbatch'` (default),
**generates and submits a wrapper sbatch** that mirrors the proven `_primus_405b_srun.sbatch`.

```
Cursor ──tools/call primus_submit_train(exp_config, nodelist=r16-[…], steps=10, fp8, …)──► server.py
server.py composes MaxText overrides:  steps=10 dataset_type=synthetic max_target_length=8192
                                        quantization=fp8 per_device_batch_size=5 ici_fsdp=8 …
server.py generates wrapper sbatch (base64-shipped to logs/_gen_<job>.sbatch):
   #SBATCH -N 8  -p <derived>  --nodelist=<nodes>
   export NCCL/GLOO_SOCKET_IFNAME=enp81s0f1
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.93          # cap HBM pool, leave room for RCCL
   srun … 'docker rm -f primus-training'                # pre-clean stale containers
   srun … 'wait for /dev/kfd,/dev/dri,/dev/infiniband'  # device-settle (post-cancel reset race)
   srun --ntasks-per-node=1 -N 8 --kill-on-bad-exit=1 \
        bash runner/primus-cli-slurm-entry.sh --config runner/use_ainic.yaml -- train pretrain …
server.py: sbatch <wrapper>  ──►  Submitted batch job <id>  ─────────────────────────────► Cursor
```

`launcher='srun'` is the direct blocking path via `primus_cli()` (no wrapper). `dry_run=True`
returns the generated script without submitting.

### Partition auto-derivation

```python
def partition_from_nodelist(nodelist: str) -> str:
    m = re.search(r"-(r\d+)-", nodelist)         # smci355-ccs-aus-r16-… -> r16
    return f"mi355x-{m.group(1)}" if m else ""    # -> mi355x-r16
```

| Nodelist | Derived partition |
|----------|-------------------|
| `smci355-ccs-aus-r16-[06,…]` | `mi355x-r16` |
| `smci355-ccs-aus-r17-[06,…]` | `mi355x-r17` |

---

## 7. Where the correctness fixes live (outside the MCP)

The MCP is a launcher; the fixes that make 405B/70B actually train are **on disk in `$PRIMUS_DIR`**
and inherited by every submission:

- `runner/use_ainic.yaml` — ionic **native-IB-verbs** RCCL env (`NCCL_IB_HCA=ionic_0..8`, TC/FIFO
  tuning, DMA-BUF) instead of the ANP-plugin default.
- `runner/helpers/hooks/train/pretrain/maxtext/prepare.py` — corrected `XLA_FLAGS`
  (latency-hiding scheduler **off**, pipelined all-gather/all-reduce/reduce-scatter **off**, no
  8 GiB combine thresholds) that otherwise deadlock 64-GPU RoCE rings.
- `runner/primus-cli-container.sh` — env-forward allowlist (`XLA_|JAX_|HSA_|…`) + sudo-docker
  detection.

---

## 8. Safety guardrails

- Write tools are narrow and **verify state first**: `slurm_cancel_job` confirms the job is in
  `squeue` before `scancel`.
- The generated wrapper **pre-cleans and waits for device settle** so an abrupt prior cancel's GPU
  reset can't fail the next launch on a random node.
- `dry_run=True` is the default on submit-style tools — the agent previews the assembled command
  before anything runs.
- The server runs as the user's own account, inheriting their Slurm/SSH permissions and active
  Conductor reservation.

---

## 9. Deployment & the reload requirement

The server lives at `/mnt/dcgpuval/afde/sshetkud/primus-mcp/` with its own `.venv`
(`mcp` / FastMCP, pinned `<2.0.0`). Files edited from Windows are CRLF-stripped on the host.

> **Reload rule:** `server.py` registers every `@_tool` in memory at import time, then `mcp.run()`
> blocks and serves that registry. Editing the file on disk does **not** hot-reload the running
> process, and Cursor caches the `tools/list` handshake. **Any change to `server.py` (new tools,
> changed signatures/logic) requires a Cursor MCP reload** to respawn the process and re-register.
> Changes to data the tools read at call time (the Primus tree, sbatch scripts, Slurm state, logs)
> do **not** need a reload.

**Design trade-offs**

| Choice | Benefit | Cost |
|--------|---------|------|
| Typed knobs → generated wrapper | Reproduces the validated 7314 launch exactly; no free-form drift | Wrapper logic lives in the tool, changes need a reload |
| Launcher-only (fixes on disk) | Any primus-cli path inherits the fixes | MCP correctness depends on `$PRIMUS_DIR` pointing at the fixed tree |
| Stateless text I/O | Simple, agent-friendly | Agent parses text, not structured results |
| Runs as the user on the jump host | No extra auth plumbing | Inherits full user permissions — hence verify-first discipline |

---

## 10. Related docs

- [lab-gpu-mcp-server-design.md](lab-gpu-mcp-server-design.md) — the vLLM/ATOM serving MCP this
  design mirrors.
- [vllm-grafana-architecture.md](vllm-grafana-architecture.md) — monitoring stack for serving runs.
