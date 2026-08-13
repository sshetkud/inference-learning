# lab-gpu MCP — Where the Scripts Live

The `lab-gpu` MCP scripts live in **two places**: the editable source in the workspace, and the
deployed copy on the Conductor jump host where the server actually runs.

## 1. Source (workspace) — `mcp-lab/`

`c:\...\Documents\lab\Webproject\mcp-lab\`

| File | Purpose |
|------|---------|
| `server.py` | FastMCP server (all lab-gpu tools) |
| `backend.py` | Cluster I/O helpers (`ssh_run`, `srun_remote`, log/metrics) |
| `deploy.sh` | Copies to the jump host, strips CRLF, builds the `.venv` |
| `run_mcp.sh` / `setup_remote.sh` | Launch / remote-setup helpers |
| `test_remote.py` | Connectivity test |
| `list_vllm_today.sh` | Utility |
| `requirements.txt`, `README.md` | Deps + docs |

## 2. Deployed (jump host, where it actually runs) — `kimi-k3-bench/mcp-lab/`

`/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/mcp-lab/`

- `server.py` (~30 KB, updated Aug 12), `backend.py`, `run_mcp.sh`, `setup_remote.sh`,
  `test_remote.py`, `requirements.txt`
- `.venv/` — the Python env (with the `mcp` SDK) that runs it
- plus `server.py.bak.*` backups

Cursor launches it via `.cursor/mcp.json`, which SSHes to the jump host and execs:

```bash
/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/mcp-lab/.venv/bin/python \
  /mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/mcp-lab/server.py
```

## Note: different layout from `primus-gpu`

`primus-gpu` lives at `/mnt/dcgpuval/afde/sshetkud/primus-mcp/` on the jump host (no separate
workspace copy). Its design write-up is at
[primus-gpu-mcp-server-design.md](primus-gpu-mcp-server-design.md) and its source snapshot at
[primus-gpu-server.py](primus-gpu-server.py).

## Related docs

- [lab-gpu-mcp-server-design.md](lab-gpu-mcp-server-design.md) — full design & architecture of the
  lab-gpu MCP server.
