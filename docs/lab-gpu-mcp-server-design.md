# lab-gpu MCP Server — Design & Architecture

How the **`lab-gpu`** Model Context Protocol (MCP) server is designed, deployed, and wired so a
Cursor agent can drive Kimi-K3 **vLLM / ATOM** Slurm jobs on AMD MI355X (`mi355x-r16`,
`mi355x-r17`) from the Conductor jump host — submit, benchmark, inspect, and cancel — without
ever touching the Grafana/Prometheus monitoring stack.

> **Key framing:** The MCP server process runs on the **Conductor jump host** (`smc200x`), not
> on the laptop and not on the GPU node. Cursor connects to it over **SSH stdio**. Because the
> server lives on the jump host, its tools call `squeue` / `sbatch` / `scancel` / `srun` as plain
> local subprocesses and reach into running jobs via `srun --overlap` — no nested SSH, no
> per-call re-auth. It is **read-mostly**: it never restarts Grafana, Prometheus, or the exporter.

---

## 1. What runs where

| Location | Component | Container? | Port |
|----------|-----------|------------|------|
| **Laptop** | Cursor (MCP client) | No | — |
| **Jump host** (`smc200x`) | **`server.py`** (FastMCP server, in `.venv`) | No — host process | stdio (over SSH) |
| **Jump host** | `backend.py` command helpers (`ssh_run`, `srun_remote`) | No | — |
| **GPU node** (e.g. `r16-10`, `r17-10`) | Slurm job → Docker (`kimi_k3` vLLM / `atom_job`) | Yes | 8000 (`/v1/models`, `/metrics`) |
| **Jump host** | Grafana / Prometheus / `vllm_exporter` (**out of scope — never touched**) | Yes | 3001 / 9091 / 9401 |

NFS (`/mnt/dcgpuval/...`) is shared, so job logs and benchmark JSONs are read directly on the
jump host without SSH to the compute node.

---

## 2. Transport & topology

```
Cursor (laptop)
   │  MCP stdio (JSON-RPC 2.0) tunneled over SSH
   ▼
server.py  (jump host smc200x, .venv/bin/python)
   │  local subprocess: squeue / sbatch / scancel / tail / curl / ssh-to-node
   │  srun --jobid=<id> --overlap …           (reach inside a running allocation)
   ▼
Slurm controller · compute nodes (vLLM :8000 / atom :8000) · NFS logs · model weights
```

`.cursor/mcp.json` launches the server per session:

```json
{
  "mcpServers": {
    "lab-gpu": {
      "command": "ssh",
      "args": [
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
        "sshetkud@smc200x-ccs-e12-31.cs-aus.dcgpu",
        "bash", "-lc",
        "export LAB_MCP_LOCAL=1; exec /mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/mcp-lab/.venv/bin/python /mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/mcp-lab/server.py"
      ]
    }
  }
}
```

Cursor spawns **one** `server.py` process and keeps the stdio pipe open for the whole session.

---

## 3. Two-layer design: transport vs. cluster I/O

The server is deliberately split so tools stay tiny and portable.

| Layer | File | Responsibility |
|-------|------|----------------|
| **Tool surface** | `server.py` | Declares MCP tools/resources, validates args, formats results. No knowledge of *how* commands run. |
| **Cluster I/O** | `backend.py` | Runs commands locally or over SSH, wraps `srun`, tails NFS logs, greps exporter metrics. Auto-detects local vs. remote mode. |

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
| `srun_remote(job_id, inner)` | Run `inner` on a job's compute node: `srun --jobid=<id> --overlap bash -lc <inner>`. |
| `tail_vllm_log(job_id, lines)` | `tail` the NFS log `kimi-k3-vllm-<id>.log`. |
| `exporter_metrics_grep(job_id)` | Read-only grep of `vllm_exporter` `:9401/metrics` (never restarts it). |

Config is environment-driven (no hardcoded paths):

| Var | Default |
|-----|---------|
| `LAB_JUMP_HOST` | `sshetkud@smc200x-ccs-e12-31.cs-aus.dcgpu` |
| `LAB_BENCH_DIR` | `/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench` |
| `LAB_LOG_DIR` | `/mnt/dcgpuval/afde/sshetkud` |
| `LAB_MCP_LOCAL` | `1` (forces local/subprocess mode on the jump host) |

---

## 4. Tool pattern

Every tool is a plain, type-hinted Python function; FastMCP turns the signature into the JSON
schema the agent sees. A shared decorator enforces three conventions.

```python
def _tool(fn):
    """Wrap tool handlers so SSH/Slurm errors return text instead of crashing MCP."""
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

1. **Return text, never raise.** Any failure becomes a readable string the agent can reason about,
   instead of tearing down the MCP session.
2. **Validate + sanitize inputs.** Job IDs must be numeric (`jid.isdigit()`); every value
   interpolated into a shell string is `shlex.quote(...)`-escaped to prevent injection.
3. **Defensive shell.** Commands use `|| true` / `|| echo 'not in queue'` so "nothing found" is a
   normal result, not an error.

---

## 5. Tool & resource catalog

### Read / status

| Tool | Args | Action |
|------|------|--------|
| `lab_info` | — | Jump host, bench dir, execution mode, Grafana safety note |
| `slurm_list_vllm_jobs` | — | List running/pending `kimi-k3-vllm*` jobs |
| `slurm_list_atom_jobs` | — | List running/pending `atom_job` jobs |
| `slurm_job_status` | `job_id` | `squeue` + `sacct` summary for one job |

### Inspect a running job

| Tool | Args | Action |
|------|------|--------|
| `vllm_models` | `job_id` | `GET /v1/models` via `srun --overlap curl` |
| `vllm_chat` | `job_id, prompt, max_tokens` | One `/v1/chat/completions` request |
| `vllm_benchmark` | `job_id, concurrency, num_prompts, input_len, output_len` | Run one `vllm bench serve` level (via `srun` + `docker exec kimi_k3`) |
| `vllm_log_tail` | `job_id, lines` | Tail NFS `kimi-k3-vllm-<id>.log` (read-only) |
| `vllm_metrics_snapshot` | `job_id` | Grep `vllm_exporter` metrics (read-only) |

### Act (write)

| Tool | Args | Action |
|------|------|--------|
| `slurm_submit_vllm_job` | `node, mode, partition` | Preflight + `sbatch --partition=<p> --nodelist=<node> run_kimi_k3_vllm-*.sbatch` |
| `slurm_submit_atom_job` | `node, mode, max_model_len, max_num_seqs, block_size` | Preflight + `sbatch --nodelist=<node> atom_job.slurm` via `atom_agent.py` |
| `slurm_cancel_job` | `job_id` | Verify in queue, then `scancel <job_id>` |

### Resources (read-only, URI-addressable)

| URI | Contents |
|-----|----------|
| `log://vllm/{job_id}` | Last 100 lines of the NFS job log |
| `metrics://vllm/{job_id}` | `vllm_exporter` snapshot for the job |

---

## 6. Example call flows

### `vllm_benchmark` (Cursor → jump host → compute node)

```
Cursor  ──tools/call vllm_benchmark(job_id=6877, c=4, isl=1024, osl=128)──►  server.py
server.py builds: vllm bench serve --model /model_weights --dataset-name random …
server.py wraps:  srun --jobid=6877 --overlap bash -lc 'sudo docker exec kimi_k3 bash -lc "<bench>"'
                     │
                     ▼
              compute node r17-10 → kimi_k3 container → vLLM :8000
                     │  (Serving Benchmark Result: throughput, TTFT/TPOT/E2E)
                     ▼
server.py returns the bench summary as text ─────────────────────────────►  Cursor
```

### `slurm_submit_vllm_job` — partition auto-derivation

The vLLM sbatch scripts hardcode `#SBATCH -p mi355x-r17`. Submitting to an **r16** node with that
partition fails with `Requested node configuration is not available`. The tool derives the
partition from the node's rack and passes `--partition` to override:

```python
def _partition_for_node(short: str) -> str:
    m = re.search(r"-(r\d+)-", short)       # smci355-ccs-aus-r16-10 -> r16
    return f"mi355x-{m.group(1)}" if m else ""   # -> mi355x-r16
```

| Node | Derived partition |
|------|-------------------|
| `smci355-ccs-aus-r16-10` | `mi355x-r16` |
| `smci355-ccs-aus-r17-14` | `mi355x-r17` |

Flow: **preflight** (`docker rm -f kimi_k3`; confirm model `config.json` on node) → if OK,
`sbatch --partition=<derived> --nodelist=<node> <script>` → return `Submitted batch job <id>`.

---

## 7. Safety guardrails

The server shares a host with a live monitoring stack, so it is designed **non-destructive**:

- Server `instructions` tell the agent to never restart Grafana / Prometheus / `vllm_exporter`.
- Metrics/log tools only `curl` / `grep` / `tail`.
- Write tools are narrow and **verify state first**:
  - `slurm_cancel_job` confirms the job is in `squeue` before `scancel`.
  - Submit tools run a **preflight** (stale-container cleanup, model/VRAM/port checks) before `sbatch`.
- The server runs as the user's own account, inheriting their Slurm/SSH permissions and active
  Conductor reservation.

---

## 8. Deployment & the reload requirement

Deploy with `mcp-lab/deploy.sh` (copies to NFS, strips Windows CRLF, builds the `.venv`), pinned to
`mcp>=1.6.0,<2.0.0` (FastMCP is not in `mcp` 2.x yet).

> **Reload rule:** `server.py` registers every `@_tool` in memory at import time, then `mcp.run()`
> blocks and serves that registry. Editing the file on disk does **not** hot-reload the running
> process, and Cursor caches the `tools/list` handshake. **Any change to `server.py` (new tools,
> changed signatures/logic) requires a Cursor MCP reload** to respawn the process and re-register.
> Changes to data the tools read at call time (sbatch scripts, Slurm state, logs) do **not** need a
> reload.

**Design trade-offs**

| Choice | Benefit | Cost |
|--------|---------|------|
| One tool per action | Clear schemas, tight guardrails | Every new capability needs a code change + reload |
| Stateless text I/O | Simple, agent-friendly | No structured/typed results; agent parses text |
| Runs as the user on the jump host | No extra auth plumbing | Inherits full user permissions — hence read-mostly discipline |

---

## 9. Related docs

- [vllm-grafana-architecture.md](vllm-grafana-architecture.md) — the monitoring stack this server
  deliberately leaves untouched.
- [atom-grafana-architecture.md](atom-grafana-architecture.md) — ATOM counterpart.
- [atom-vs-vllm.md](atom-vs-vllm.md) — the two serving stacks the submit/benchmark tools drive.
