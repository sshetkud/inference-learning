# Log file locations (Kimi-K3 Slurm jobs on MI355X)

Where to find logs for the Slurm + Docker inference jobs (vLLM / SGLang / ATOM) run on the
`mi355x-r16` / `mi355x-r17` partitions through the controller
`sshetkud@smc200x-ccs-e12-31.cs-aus.dcgpu`.

All paths below live on the shared **NFS** mount `/mnt/dcgpuval/...`, so they can be read
directly from the controller (or any node) with `tail`/`cat` — **no `docker logs` needed**.

> Replace `<JID>` with the Slurm job id (e.g. `9549`).

---

## Why `docker logs <container>` is usually empty

For the multi-node and some single-node jobs the container's **main process is `sleep infinity`**;
Ray and vLLM/SGLang are started inside it via `docker exec`. `docker logs` only shows the *main*
process output (the silent `sleep`), so it looks empty. Read the NFS log files below instead, or
`docker exec <container> ...` to see live component logs.

---

## Multi-node vLLM (Ray, TP×PP) — `run_kimi_k3_vllm_8n_ray.sbatch`

| What | Path |
|------|------|
| Driver / orchestration (image pull, `pip install ray`, 64-GPU join count, ray head/worker, benchmark sweep) | `/mnt/dcgpuval/afde/sshetkud/kimi-k3-vllm-8n-<JID>.log` |
| vLLM engine (memory check, weight load, KV cache, CUDA graph, `/health`) | `/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/vllm_serve_<JID>.log` |
| Per-concurrency benchmark logs | `/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/vllm8n_bench_<JID>/bench_c{16,32,64,128,256}.log` |
| Benchmark result JSONs | `/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/vllm8n_bench_<JID>/kimi_k3_vllm8n_c{...}_i8192_o1024.json` |

### Ray cluster coordination state

| What | Path |
|------|------|
| Head node IP (once head is up) | `/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/ray_head_<JID>.ip` |
| Workers joined (one file per worker) | `/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/ray_ready_<JID>/*.joined` |
| Done flag (appears when benchmark finishes) | `/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/ray_done_<JID>.flag` |

### Live Ray / vLLM from inside the head container (r16-06)

```bash
sudo docker exec kimi_k3_ray ray status                                   # live 64-GPU cluster
sudo docker exec kimi_k3_ray bash -lc 'tail -f /tmp/ray/session_latest/logs/*'
```

---

## Single-node SGLang — `run_kimi_k3_sglang-*.sbatch`

| What | Path |
|------|------|
| Driver + engine (combined) | `/mnt/dcgpuval/afde/sshetkud/kimi-k3-sglang-<JID>.log` |
| Per-concurrency benchmark logs | `/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/sglang_bench_<JID>/bench_c{2,4,8,16,32}.log` |
| Benchmark result JSONLs | `/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/sglang_bench_<JID>/kimi_k3_sglang_c{...}_i8192_o1024.jsonl` |

Container name: `sglang_job` (live engine: `sudo docker logs -f sglang_job`, since it runs in the foreground).

---

## Single-node vLLM — `run_kimi_k3_vllm*.sbatch`

| What | Path |
|------|------|
| Driver + engine (combined) | `/mnt/dcgpuval/afde/sshetkud/kimi-k3-vllm-<JID>.log` |
| Per-concurrency benchmark logs | `/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/bench_<JID>_c{16,32,64,128}.log` |
| Benchmark result JSONs | `/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/kimi_k3_vllm_serving_c{...}.json.<JID>` |

Container name: `kimi_k3`.

---

## Single-node ATOM — `atom_job.slurm`

| What | Path |
|------|------|
| Serve logs (node-local on compute node) | `/tmp/atom_job-<JID>.out` / `.err` |
| Benchmark result JSONs | `/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/atom_bench_<JID>/kimi_k3_atom_c{...}_i8192_o1024.json` |
| Per-concurrency benchmark logs | `/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/atom_bench_<JID>/bench_c{...}.log` |
| Watcher log | `/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/watch_bench_<JID>.log` |

Container name: `atom_job`.

---

## Handy commands (run on the controller)

```bash
# follow driver + engine logs
tail -f /mnt/dcgpuval/afde/sshetkud/kimi-k3-vllm-8n-<JID>.log
tail -f /mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/vllm_serve_<JID>.log

# how many Ray workers have joined
ls /mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/ray_ready_<JID>/

# job state / accounting
squeue -j <JID> -o '%.10i %.20j %.8T %.10M %.6D'
sacct -j <JID> --format=JobID,JobName%18,State,Elapsed,ExitCode

# live logs from inside a container on a compute node (compute nodes are not SSH-able directly)
srun --jobid=<JID> --overlap -N1 -w <node> bash -lc 'sudo docker exec <container> ray status'
```

---

See also: [multinode-vllm-ray.md](multinode-vllm-ray.md) for the exact sbatch + Ray launcher scripts used to run the 8-node job.
