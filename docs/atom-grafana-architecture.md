# Kimi-K3 ATOM Grafana — Full Dashboard Architecture

How the **ATOM** Grafana monitoring stack is built, deployed, and wired to live Kimi-K3
Slurm jobs on AMD MI355X (`mi355x-r17`). This was the **original** monitoring stack; the
[vLLM dashboards](./vllm-grafana-architecture.md) were later forked from these templates via
`gen_dashboards.mjs`.

> **Key framing:** The workload container (`atom_job`) runs on the **GPU compute node**.
> Grafana, Prometheus, and the collector run on the **Conductor jump host** (`smc200x`).
> There is **no monitoring sidecar** on the compute node — the jump host reaches the job via
> `srun` (and occasionally SSH fallback) plus shared NFS for benchmark artifacts.

---

## 1. What runs where

| Location | Component | Container? | Port |
|----------|-----------|------------|------|
| **GPU node** (e.g. `r17-14`) | Slurm job → Docker **`atom_job`** (ATOM OpenAI server) | Yes | 8000 (`/metrics`, `/v1/models`) |
| **GPU node** | Serve stdout/stderr | — | **Node-local** `/tmp/atom_job-<id>.out` / `.err` |
| **Jump host** (`smc200x`) | **`atom_exporter.py`** (Python collector) | No — host process | 9400 |
| **Jump host** | Prometheus (`atom-k3-prometheus`) | Yes | 9091 |
| **Jump host** | Grafana (`atom-k3-grafana`) | Yes | 3000 (internal) |
| **Jump host** | nginx proxy (`atom-k3-proxy`) | Yes | **3001** (browser) |

NFS bench dir holds **watcher log** and **benchmark JSONs**. Serve logs stay on the compute
node (NFS is often full on `/dcgpuval`).

---

## 2. End-to-end data flow

```
Compute node (r17-XX)              Jump host (smc200x)
┌─────────────────────┐            ┌──────────────────────────────────┐
│ atom_job :8000      │            │ atom_exporter.py  :9400/metrics  │
│  /metrics (vllm:*)  │◄── srun ───│   polls /metrics + NFS JSONs     │
│  /v1/models         │    curl    │   watch_bench_<id>.log on NFS    │
│ /tmp/atom_job-*.err │◄─ srun ───│   tail serve .out/.err via srun  │
└─────────────────────┘            └──────────┬───────────────────────┘
         │                                    │ scrape every 15s
         │ NFS (watcher + bench JSONs)        ▼
         └──────────────────────────────────► Prometheus :9091
                                              └──► Grafana :3001 (folder: ATOM)
```

### What the exporter collects (every ~10 s per `--job-id`)

| Metric family | Source |
|---------------|--------|
| `atom_serving_*` | Live ATOM `/metrics` via `srun --jobid=<id> --overlap curl localhost:8000/metrics` (parsed `vllm:*` series) |
| `atom_server_up` | `srun … curl localhost:8000/v1/models` |
| `atom_bench_*` | NFS `atom_bench_<id>/kimi_k3_atom_c{C}_i8192_o1024.json` |
| `atom_job_phase`, `atom_watcher_current_concurrency` | NFS `watch_bench_<id>.log` |
| `atom_watcher_log_bytes` | Size of watcher log on NFS |
| `atom_slurm_*` | `squeue -j <id>` on jump host |
| `atom_serve_config_info` | Static ATOM serve flags + tunables from `serve_config_<id>.json`, parsed sbatch output, or defaults |
| `atom_exporter_up`, `atom_job_active` | Exporter self-metrics |

When live `/metrics` is unavailable (model still loading), some serving panels can fall back to
the latest completed benchmark JSON (`serving_metrics_source` label: `prometheus` | `bench` | `none`).

Prometheus scrapes `host.docker.internal:9400`. Grafana uses datasource `atom-prom` →
`http://prometheus:9090`.

---

## 3. How the dashboards were created

Unlike vLLM (generated from ATOM), **ATOM dashboards are hand-authored JSON** — the source
templates for the whole Kimi-K3 monitoring UI.

### Dashboard files

| File | Grafana UID | Tab title |
|------|-------------|-----------|
| `atom-kimi-k3.json` | `atom-kimi-k3-run` | **ATOM Execution** |
| `atom-kimi-k3-benchmark.json` | `atom-kimi-k3-benchmark` | **Benchmark** |
| `atom-kimi-k3-monitor.json` | `atom-kimi-k3-monitor` | **Job Monitor** |

Location: `.cursor/skills/atom-kimi-k3/grafana/dashboards/`

Each panel is a PromQL query against `atom_*` metrics, filtered by template variable `$job_id`:

```promql
label_values(atom_exporter_up, job_id)
```

### Provisioning (file-based, preferred)

Grafana auto-loads from NFS via:

```yaml
# grafana/provisioning/dashboards/default.yml
providers:
  - name: atom-kimi-k3
    folder: ATOM
    type: file
    updateIntervalSeconds: 30
    options:
      path: /etc/grafana/dashboards
```

vLLM dashboards are loaded separately into folder **vLLM** via `vllm.yml` (same Docker stack).

### Alternative: API import

For one-off updates without restarting Grafana:

```bash
python3 grafana/import_dashboards.py http://127.0.0.1:3001
```

Posts each JSON to Grafana's `/api/dashboards/db` with `overwrite: true`.

### Downstream: vLLM fork

`vLLM-kimi-k3/grafana/gen_dashboards.mjs` reads these three ATOM JSON files, renames metrics
and URLs, and writes `vllm-kimi-k3-*.json`. ATOM remains the canonical dashboard design.

---

## 4. Dashboard tabs (folder **ATOM**)

| Tab | UID | Primary data |
|-----|-----|--------------|
| **ATOM Execution** | `atom-kimi-k3-run` | Live `atom_serving_*`, phase timeline, serve-config table, embedded submit form |
| **Benchmark** | `atom-kimi-k3-benchmark` | `atom_bench_*` TPS / TTFT / TPOT bar charts + results table |
| **Job Monitor** | `atom-kimi-k3-monitor` | Slurm status, watcher progress, serve stderr iframe, live serving when up |

### ATOM Execution highlights

- **Server UP/DOWN**, phase stat, live req/s & gen tok/s, KV cache, exporter UP
- **Server config (sbatch)** table from `atom_serve_config_info{job_id="$job_id"}`
- **Phase timeline:** loading → ready → benchmarking → done
- **Live serving:** throughput, queue, TTFT p50, E2E latency
- **Submit form** (or link to `/atom/submit`) with tunable `max-model-len`, `max-num-seqs`, `block-size`

### Benchmark tab

Populates as JSON files appear under `atom_bench_<id>/`:

```
kimi_k3_atom_c{C}_i8192_o1024.json   # C ∈ 1, 4, 8, 16, 32, 64, 128
```

---

## 5. nginx routing (non-metric UI)

Browser hits **nginx :3001**:

| URL | Handler | Purpose |
|-----|---------|---------|
| `/` | Grafana | Dashboards |
| `/atom/submit` | `atom_exporter.py` | HTML form → preflight → `sbatch atom_job.slurm` |
| `/atom/api/preflight?node=…` | `atom_exporter.py` | JSON preflight (VRAM, stale containers, port 8000) |
| `/atom/monitor-embed?job_id=…` | `atom_exporter.py` | Live log iframe (serve .out/.err + watcher) |

Job Monitor embeds `/atom/monitor-embed` in HTML iframes. The embed shows:

1. **Serve stderr** — model load progress (`/tmp/atom_job-<id>.err`, tail via srun)
2. **Serve stdout** — server startup lines (`.out`)
3. **Watcher log** — benchmark sweep progress on NFS (`watch_bench_<id>.log`)

---

## 6. Slurm job architecture (what dashboards monitor)

ATOM uses a **two-process** pattern (unlike vLLM's single integrated sbatch):

| Process | Slurm job name | Role |
|---------|----------------|------|
| **Serve** | `atom_job` | Docker `atom_job` container, ATOM OpenAI server |
| **Watcher** | (separate script in `atom_job.slurm`) | Polls server ready, runs benchmark sweep, writes NFS JSONs |

Key paths:

| Artifact | Location |
|----------|----------|
| Serve stdout/stderr | **Node-local** `/tmp/atom_job-<id>.out` / `.err` |
| Watcher log | NFS `watch_bench_<id>.log` |
| Bench results | NFS `atom_bench_<id>/` |
| Serve config snapshot | NFS `serve_config_<id>.json` (from submit form) |
| Sbatch script | `atom_job.slurm` in bench dir |

Default serve image: `rocm/atom-dev:rocm7.2.4_ubuntu24.04_py3.12_pytorch2.10.0_20260727_kimi_k3`

---

## 7. Deploy and bring-up

### Cluster paths

```
/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/
├── atom_exporter.py
├── atom_grafana_up.sh
├── atom_agent.py
├── atom_metrics_parser.py   # shared with vLLM exporter
└── grafana/
    ├── docker-compose.yml
    ├── prometheus.yml         # scrapes :9400 (ATOM) and :9401 (vLLM)
    ├── nginx.conf
    ├── dashboards/            # ATOM JSON dashboards
    ├── dashboards-vllm/         # vLLM JSON dashboards (separate folder)
    └── provisioning/
```

### Start monitoring (jump host)

```bash
bash /mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/atom_grafana_up.sh <SERVE_JOB_ID>
```

Or via agent:

```bash
python3 .../atom_agent.py monitor <SERVE_JOB_ID>
```

`atom_grafana_up.sh`:

1. Stops prior exporter on port 9400
2. `chmod -R a+rX` provisioning + dashboards (Grafana container runs as uid **472**)
3. `docker compose up -d`
4. Starts `python3 atom_exporter.py --job-id <id> … --poll-interval 10`
5. Auto-discovers additional jobs from `squeue -n atom_job` and NFS `atom_bench_*/`

### Browser access

```bash
ssh -L 3001:localhost:3001 sshetkud@smc200x-ccs-e12-31.cs-aus.dcgpu
# http://localhost:3001/d/atom-kimi-k3-run/kimi-k3-atom-run?var-job_id=<id>
```

Login: `admin` / `admin`

### Deploy from laptop

```bash
scp -r .cursor/skills/atom-kimi-k3/grafana \
    .cursor/skills/atom-kimi-k3/scripts/atom_*.py \
    sshetkud@smc200x-ccs-e12-31.cs-aus.dcgpu:/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/
```

Ensure scripts use LF line endings (`fix_crlf.py` or `sed -i 's/\r$//'` after Windows upload).

---

## 8. Opening a dashboard — step by step

1. Browser → **nginx :3001** → Grafana renders JSON from folder **ATOM**
2. Select **Slurm job** from dropdown (jobs with `atom_exporter_up` metric)
3. Grafana runs PromQL (e.g. `atom_serving_generation_tokens_per_sec{job_id="5954"}`)
4. Prometheus returns values scraped from **atom_exporter :9400**
5. Exporter gathered data by:
   - `srun curl` into the Slurm allocation for `/metrics` and `/v1/models`
   - Reading NFS watcher log and `atom_bench_<id>/` JSONs
   - Running `squeue` locally
6. Log iframes load `/atom/monitor-embed`; exporter tails node-local `.out/.err` via srun
   (SSH fallback if job left queue but node is known from `sacct`)

---

## 9. Repo layout (source of truth)

```
.cursor/skills/atom-kimi-k3/
├── grafana/
│   ├── docker-compose.yml       # Shared stack (ATOM + vLLM)
│   ├── prometheus.yml
│   ├── nginx.conf
│   ├── README.md
│   ├── dashboards/              # ATOM — canonical JSON (3 tabs)
│   ├── dashboards-vllm/         # vLLM copies (generated)
│   ├── provisioning/
│   │   ├── dashboards/default.yml   # folder: ATOM
│   │   ├── dashboards/vllm.yml      # folder: vLLM
│   │   └── datasources/prometheus.yml
│   └── import_dashboards.py
└── scripts/
    ├── atom_exporter.py         # Collector + /atom/submit + /monitor-embed
    ├── atom_grafana_up.sh
    ├── atom_agent.py            # CLI: full, preflight, monitor, results
    └── atom_metrics_parser.py   # Parse vLLM-format /metrics text
```

---

## 10. ATOM vs vLLM monitoring (same stack)

| Aspect | ATOM | vLLM |
|--------|------|------|
| Exporter port | 9400 | 9401 |
| Metrics prefix | `atom_*` | `vllm_k3_*` |
| Workload container | `atom_job` | `kimi_k3` |
| Slurm job name | `atom_job` | `kimi-k3-vllm-csweep` |
| Serve logs | Node-local `/tmp/atom_job-*` | NFS `kimi-k3-vllm-<id>.log` |
| Watcher / bench | Separate watcher + `watch_bench_*.log` | Benchmark inside sbatch |
| Bench JSON path | `atom_bench_<id>/kimi_k3_atom_c*.json` | `kimi_k3_vllm_serving_c*.json.<id>` |
| Concurrency sweep | 1, 4, 8, 16, 32, 64, 128 | 16, 32, 64, 128 |
| Submit preflight | Built into `/atom/submit` + API | vLLM submit (node pick) |
| Dashboard origin | Hand-authored (canonical) | Generated from ATOM JSON |
| Grafana folder | ATOM | vLLM |
| nginx prefix | `/atom/` | `/vllm/` |

Both exporters are plain Python on the jump host; one Prometheus + Grafana stack serves both.

---

## 11. Common pitfalls

| Symptom | Cause |
|---------|-------|
| Empty dashboard / no datasource | NFS `grafana/provisioning/` or `dashboards/` not world-readable — Grafana uid 472 cannot read them. Fix: `chmod -R a+rX …` and restart Grafana |
| Empty benchmark panels | Wrong job selected, or serve failed before watcher wrote JSONs |
| Empty serving during load | Expected — wait until `atom_server_up=1` and `/metrics` responds |
| Serve log iframe empty | Job ended and node unknown — exporter falls back to SSH; may fail if node unreachable |
| Exporter crash after Windows upload | CRLF in `atom_exporter.py` / shell scripts — strip `\r` |
| Benchmark shows zeros | Job like 5509 with failed bench — pick a job with valid `atom_bench_<id>/` JSONs |

---

## Related docs

- [ATOM vs vLLM — Architecture & Kernels](./atom-vs-vllm.md)
- [Kimi-K3 vLLM Grafana architecture](./vllm-grafana-architecture.md)
- ATOM skill: `.cursor/skills/atom-kimi-k3/SKILL.md`
- Grafana README: `.cursor/skills/atom-kimi-k3/grafana/README.md`
- [ROCm MAD Kimi-K3 ATOM benchmark](https://github.com/ROCm/MAD/blob/develop/benchmark/kimi_k3/README.md)
