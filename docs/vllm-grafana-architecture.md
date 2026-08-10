# Kimi-K3 vLLM Grafana — Full Dashboard Architecture

How the **vLLM** Grafana monitoring stack is built, deployed, and wired to live Kimi-K3
Slurm jobs on AMD MI355X (`mi355x-r17`). Shares the same jump-host stack as ATOM Kimi-K3
monitoring; vLLM adds its own exporter, metrics prefix, and dashboard folder.

> **Key framing:** The workload container (`kimi_k3`) runs on the **GPU compute node**.
> Grafana, Prometheus, and the collector run on the **Conductor jump host** (`smc200x`).
> There is **no monitoring sidecar** on the compute node — the jump host reaches the job via
> `srun` and shared NFS.

---

## 1. What runs where

| Location | Component | Container? | Port |
|----------|-----------|------------|------|
| **GPU node** (e.g. `r17-14`) | Slurm job → Docker **`kimi_k3`** (vLLM OpenAI server) | Yes | 8000 (`/metrics`, `/v1/models`) |
| **Jump host** (`smc200x`) | **`vllm_exporter.py`** (Python collector) | No — host process | 9401 |
| **Jump host** | Prometheus (`atom-k3-prometheus`) | Yes | 9091 |
| **Jump host** | Grafana (`atom-k3-grafana`) | Yes | 3000 (internal) |
| **Jump host** | nginx proxy (`atom-k3-proxy`) | Yes | **3001** (browser) |

NFS (`/mnt/dcgpuval/...`) is shared: job logs and benchmark JSONs are read directly on the
jump host without SSH to the compute node.

---

## 2. End-to-end data flow

```
Compute node (r17-XX)              Jump host (smc200x)
┌─────────────────────┐            ┌──────────────────────────────────┐
│ kimi_k3 container   │            │ vllm_exporter.py  :9401/metrics  │
│  :8000/metrics      │◄── srun ───│   polls /metrics + NFS JSONs    │
│  /v1/models         │    curl    │   squeue, job log, docker logs  │
└─────────────────────┘            └──────────┬───────────────────────┘
         │                                    │ scrape every 15s
         │ NFS (shared)                       ▼
         └──────────────────────────────────► Prometheus :9091
                                              └──► Grafana :3001 (folder: vLLM)
```

### What the exporter collects (every ~15 s per `--job-id`)

| Metric family | Source |
|---------------|--------|
| `vllm_k3_serving_*` | Live vLLM native `/metrics` via `srun --jobid=<id> --overlap curl localhost:8000/metrics` |
| `vllm_k3_server_up` | `srun … curl localhost:8000/v1/models` |
| `vllm_k3_bench_*` | NFS JSONs `kimi_k3_vllm_serving_c{C}.json.<id>` |
| `vllm_k3_job_phase`, `vllm_k3_current_concurrency` | Parse NFS log `kimi-k3-vllm-<id>.log` |
| `vllm_k3_slurm_*` | `squeue -j <id>` on jump host |
| `vllm_k3_job_start_timestamp` | First `[timestamp] JOB=…` line in NFS log (or `sacct Start`) |
| `vllm_k3_server_ready_timestamp` | Wall time when `/v1/models` first succeeds |
| `vllm_k3_serve_config_info` | Static sbatch serve flags (+ optional per-job overrides) |

Prometheus scrapes `host.docker.internal:9401`. Grafana panels run PromQL against datasource
`atom-prom` → `http://prometheus:9090`.

---

## 3. How the dashboards were created

### Step A — Fork from ATOM templates

Three working ATOM Grafana JSON files were the starting point:

| ATOM source | vLLM output |
|-------------|-------------|
| `atom-kimi-k3.json` | `vllm-kimi-k3-run.json` |
| `atom-kimi-k3-benchmark.json` | `vllm-kimi-k3-benchmark.json` |
| `atom-kimi-k3-monitor.json` | `vllm-kimi-k3-monitor.json` |

### Step B — Generate with `gen_dashboards.mjs`

```bash
node .cursor/skills/vLLM-kimi-k3/grafana/gen_dashboards.mjs
```

The generator:

1. Reads ATOM dashboard JSON from `atom-kimi-k3/grafana/dashboards/`
2. Bulk-replaces metric names (`atom_serving_*` → `vllm_k3_serving_*`, etc.)
3. Rewrites URLs (`/atom/submit` → `/vllm/submit`, monitor embed paths)
4. Patches panel text (container `kimi_k3`, NFS log names, vLLM-specific copy)
5. Injects extra panels (e.g. **Server config** table on the Run tab)
6. Writes output to `vLLM-kimi-k3/grafana/dashboards/`

Manual edits (timeline annotations, serve-config tables, “vLLM server” labels) are applied
directly to the generated JSON afterward.

### Step C — Grafana file provisioning

Grafana auto-loads dashboards from NFS via provisioning config:

```yaml
# grafana/provisioning/dashboards/vllm.yml
providers:
  - name: vllm-kimi-k3
    folder: vLLM
    type: file
    updateIntervalSeconds: 30
    options:
      path: /etc/grafana/dashboards-vllm
```

ATOM dashboards live in folder **ATOM** via a separate provider. Both share one Grafana
instance and one Prometheus.

---

## 4. Dashboard tabs (folder **vLLM**)

| Tab | UID | Primary data |
|-----|-----|--------------|
| **vLLM Run** | `vllm-kimi-k3-run` | Live `vllm_k3_serving_*`, serve-config table |
| **Benchmark** | `vllm-kimi-k3-benchmark` | `vllm_k3_bench_*` from NFS JSONs |
| **Job Monitor** | `vllm-kimi-k3-monitor` | Slurm state, phase, logs, annotations, live + bench |

The **Slurm job** dropdown is a Grafana template variable:

```promql
label_values(vllm_k3_exporter_up, job_id)
```

Panels filter with `{job_id="$job_id"}`. Refresh is typically **10 s** on dashboards;
Prometheus scrape interval is **15 s**.

### Timeline annotations (Job Monitor + Run)

- **Blue — Job started:** `vllm_k3_job_start_timestamp` (`useValueForTime: true`)
- **Green — Server ready:** `vllm_k3_server_ready_timestamp`

Phase/benchmark panels track from job start; live tok/s charts only populate after the green
annotation (model load often takes 1–3 h for Kimi-K3).

---

## 5. nginx routing (non-metric UI)

Browser hits **nginx :3001**. Special paths proxy to the exporter, not Grafana:

| URL | Handler | Purpose |
|-----|---------|---------|
| `/` | Grafana | Dashboards |
| `/vllm/submit` | `vllm_exporter.py` | HTML form → `sbatch` new job |
| `/vllm/monitor-embed?job_id=…` | `vllm_exporter.py` | Live log iframe (NFS tail + `docker logs` via srun) |

Job Monitor embeds `/vllm/monitor-embed` in HTML iframes inside text panels — that is how
`sudo docker logs kimi_k3` appears in Grafana without a sidecar on the compute node.

---

## 6. Deploy and bring-up

### Cluster paths

```
/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/
├── vllm_exporter.py
├── vllm_grafana_up.sh
└── grafana/
    ├── docker-compose.yml      # Prometheus + Grafana + nginx
    ├── prometheus.yml          # scrapes :9400 (ATOM) and :9401 (vLLM)
    ├── nginx.conf
    ├── dashboards-vllm/        # vLLM JSON dashboards
    └── provisioning/
```

### From laptop (after editing repo files)

```bash
scp .cursor/skills/vLLM-kimi-k3/scripts/vllm_exporter.py \
    .cursor/skills/vLLM-kimi-k3/scripts/vllm_grafana_up.sh \
    .cursor/skills/vLLM-kimi-k3/grafana/dashboards/*.json \
    sshetkud@smc200x-ccs-e12-31.cs-aus.dcgpu:/tmp/

ssh sshetkud@smc200x-ccs-e12-31.cs-aus.dcgpu \
  "bash /tmp/deploy_vllm_grafana.sh <JOB_ID>"
```

### On jump host

```bash
bash /mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/vllm_grafana_up.sh <JOB_ID>
```

This script:

1. Copies dashboard JSONs into `grafana/dashboards-vllm/`
2. Copies `vllm_exporter.py` to the bench dir (strips CRLF)
3. Runs `docker compose up -d` (Grafana stack)
4. Kills any old exporter, starts `python3 vllm_exporter.py --job-id <id> …` on port 9401
5. Waits until `/metrics` responds

### Browser access

```bash
ssh -L 3001:localhost:3001 sshetkud@smc200x-ccs-e12-31.cs-aus.dcgpu
# http://localhost:3001/d/vllm-kimi-k3-monitor/kimi-k3-vllm-job-monitor?var-job_id=<id>
```

Default Grafana login: `admin` / `admin`.

---

## 7. Opening a dashboard — step by step

1. Browser → **nginx :3001** → Grafana renders JSON from folder **vLLM**
2. Grafana runs PromQL (e.g. `vllm_k3_serving_generation_tokens_per_sec{job_id="6635"}`)
3. Prometheus returns last scraped values from **vllm_exporter :9401**
4. Exporter gathered those values by:
   - Reading NFS files (log, benchmark JSONs)
   - Running `squeue` locally
   - Running `srun curl` into the Slurm allocation to reach **`kimi_k3`** on the GPU node
5. Log iframes bypass Prometheus — browser loads `/vllm/monitor-embed`; exporter fetches
   logs on each request

---

## 8. Repo layout (source of truth)

```
.cursor/skills/
├── atom-kimi-k3/grafana/          # Shared Docker stack + ATOM dashboards
│   ├── docker-compose.yml
│   ├── prometheus.yml
│   ├── nginx.conf
│   └── provisioning/
└── vLLM-kimi-k3/
    ├── grafana/
    │   ├── gen_dashboards.mjs     # ATOM → vLLM generator
    │   ├── README.md
    │   └── dashboards/            # 3 vLLM dashboard JSONs
    └── scripts/
        ├── vllm_exporter.py       # Collector + /submit + /monitor-embed
        ├── vllm_grafana_up.sh     # Bring-up on jump host
        ├── deploy_vllm_grafana.sh
        └── vllm_agent.py          # CLI preflight + sbatch (optional)
```

---

## 9. vLLM vs ATOM monitoring (same stack)

| Aspect | ATOM | vLLM |
|--------|------|------|
| Exporter port | 9400 | 9401 |
| Metrics prefix | `atom_*` | `vllm_k3_*` |
| Workload container | `atom_job` | `kimi_k3` |
| Live serving source | ATOM `/metrics` (when available) | vLLM native `/metrics` |
| Grafana folder | ATOM | vLLM |
| nginx prefix | `/atom/` | `/vllm/` |

Both exporters run as plain Python on the jump host; both are scraped by the same Prometheus
container.

---

## 10. Common pitfalls

| Symptom | Cause |
|---------|-------|
| Empty serving charts for hours | Expected during model load — wait for green **Server ready** annotation |
| Empty benchmark panels | Job failed before benchmark (e.g. `server not ready after 180 min`) |
| Exporter down / empty panels | Restart with `vllm_grafana_up.sh`; check CRLF corruption on scripts uploaded from Windows |
| Phase charts start ~3 h before tok/s | Phase tracks job start; serving only after `/v1/models` responds |

---

## Related docs

- [ATOM vs vLLM — Architecture & Kernels](./atom-vs-vllm.md)
- vLLM Kimi-K3 skill: `.cursor/skills/vLLM-kimi-k3/SKILL.md`
- Grafana README: `.cursor/skills/vLLM-kimi-k3/grafana/README.md`
- [vLLM Kimi-K3 recipe](https://docs.vllm.ai/projects/recipes/en/stable/moonshotai/Kimi-K3.html)
