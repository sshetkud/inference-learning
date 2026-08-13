# primus-gpu MCP — Where the Scripts Live

The `primus-gpu` MCP scripts live in **one place** — directly on the Conductor jump host. Unlike
`lab-gpu`, there is no separate editable workspace copy.

## Deployed & only copy (jump host) — `primus-mcp/`

`/mnt/dcgpuval/afde/sshetkud/primus-mcp/`

| File | Purpose |
|------|---------|
| `server.py` (~24 KB) | FastMCP server — all primus-gpu tools (`primus_submit_train`, `primus_benchmark`, `primus_preflight`, `primus_projection`, `slurm_*`, `gpu_utilization`) |
| `backend.py` | Cluster I/O helpers (`ssh_run`, `primus_cli`, `partition_from_nodelist`, `tail_log_for_job`) + config constants |
| `_primus_405b_srun.sbatch` | Reference wrapper the `sbatch` path mirrors |
| `diag_env.sh`, `_fleet_check.sh` | Diagnostic / fleet-check helpers |
| `README.md` | Docs |
| `logs/` | sbatch stdout/err + generated `_gen_<job>.sbatch` wrappers |
| `.venv/` | Python env (with the `mcp` SDK) that runs the server |
| `server.py.bak.*` | Timestamped backups from patches |

Key config (in `backend.py`):

| Var | Value |
|-----|-------|
| `PRIMUS_DIR` | `/mnt/dcgpuval/afde/sshetkud/Primus` (the fixed tree) |
| `PRIMUS_LOG_DIR` | `/mnt/dcgpuval/afde/sshetkud/primus-mcp/logs` |
| `PRIMUS_SOCKET_IFNAME` | `enp81s0f1` |

Cursor launches it via `.cursor/mcp.json` (on the laptop — not present on the jump host), which
SSHes to the jump host and execs:

```bash
/mnt/dcgpuval/afde/sshetkud/primus-mcp/.venv/bin/python \
  /mnt/dcgpuval/afde/sshetkud/primus-mcp/server.py
```

## Note: different layout from `lab-gpu`

`lab-gpu` has both a workspace source (`mcp-lab/`) and a deployed copy
(`/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/mcp-lab/`); see
[lab-gpu-mcp-scripts-location.md](lab-gpu-mcp-scripts-location.md). `primus-gpu` is edited in place
on the jump host.

## Related docs

- [primus-gpu-mcp-server-design.md](primus-gpu-mcp-server-design.md) — full design & architecture.
- [primus-gpu-server.py](primus-gpu-server.py) — committed source snapshot of `server.py`.
