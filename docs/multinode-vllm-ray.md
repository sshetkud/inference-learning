# Multi-node vLLM, Slurm, and Ray

How **Slurm**, **Ray**, and **vLLM** fit together for distributed inference on AMD MI355X
clusters — plus the **Kimi-K3 8-node (TP×PP)** runbook we used on Conductor.

Related docs: [log-file-locations.md](log-file-locations.md),
[vllm-grafana-architecture.md](vllm-grafana-architecture.md),
[atom-vs-vllm.md](atom-vs-vllm.md).

Official vLLM reference: [Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/).

---

## Slurm vs Ray: who does what?

They solve **different problems**. You need both only when a **single vLLM engine**
must span multiple nodes (tensor parallel and/or pipeline parallel across nodes).

```
┌─────────────────────────────────────────────────────────────┐
│  Slurm  — resource manager                                  │
│  • Allocates N nodes, GPUs, walltime                        │
│  • Exclusive reservation (--exclusive)                      │
│  • Does NOT start cross-node vLLM worker processes          │
└──────────────────────────┬──────────────────────────────────┘
                           │ one sbatch job, all nodes
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Ray  — process / runtime orchestrator (when needed)        │
│  • Ray head on node 0 (GCS on :6379)                        │
│  • Ray workers on other nodes join head                     │
│  • Exposes cluster-wide GPU pool to vLLM                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  vLLM  — inference engine                                   │
│  • vllm serve --distributed-executor-backend ray              │
│  • TP / PP workers placed by Ray across nodes               │
│  • OpenAI API on head :8000                                 │
└─────────────────────────────────────────────────────────────┘
```

| Layer | Role | What it knows |
|-------|------|----------------|
| **Slurm** | Hardware allocation | "You have r16-06…r16-34, 8 GPUs each, 16 h" |
| **Ray** | Cross-node process mesh | "64 GPUs at these IPs; start rank N here" |
| **vLLM** | Model sharding + serving | "Layers 0–40 on ranks 0–63" |

---

## When is Ray required?

vLLM supports two distributed executor backends:

| Backend | Scope | When to use |
|---------|--------|-------------|
| **`multiprocessing` (`mp`)** | **Single node** | All GPUs on one box (default). Kimi-K3 single-node: **TP=8**, no Ray. |
| **`ray`** | **Multi-node** | One logical engine whose **TP or PP spans nodes**. |

### Ray **is** required when:

- Model shards must live on **more than one node** (TP × PP > GPUs per node).
- Example: 8 nodes × 8 GPUs, one engine → `TP=8`, `PP=8`, `--distributed-executor-backend ray`.

### Ray is **not** required when:

1. **Single-node serving** — Slurm allocates 1 node; `vllm serve --tensor-parallel-size 8` uses `mp`.
2. **Multi-node replicas** — each node runs an **independent** engine; **nginx** load-balances
   (best throughput when the model fits on one node).
3. **vLLM native data-parallel** — head + headless worker containers rendezvous without Ray.

**Rule of thumb:** Slurm gives you machines; Ray wires one distributed engine across them;
replicas/nginx wire **many independent engines**.

---

## Ray components you need

Inside **one Slurm job** (never split Ray and vLLM into separate Slurm jobs — GPU conflicts):

| Component | Typical port | Purpose |
|-----------|--------------|---------|
| **Ray head (GCS)** | **6379** | Global control store; workers register here |
| **Ray dashboard** | **8265** | Debug / cluster visibility |
| **Ray client** | **10001** | Optional client connections |
| **vLLM API** | **8000** | OpenAI-compatible HTTP on head |

Head start (conceptual):

```bash
ray start --head \
  --node-ip-address=<head_cluster_ip> \
  --port=6379 \
  --num-gpus=8 \
  --dashboard-host=0.0.0.0 \
  --disable-usage-stats
```

Worker start:

```bash
ray start --address=<head_cluster_ip>:6379 \
  --num-gpus=8 \
  --disable-usage-stats
```

Then **one** `vllm serve` on the head:

```bash
vllm serve /model_weights \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 8 \
  --distributed-executor-backend ray \
  --host 0.0.0.0 --port 8000
```

**Critical:** `--node-ip-address` / `VLLM_HOST_IP` must be the **cluster-routable**
fabric IP (not localhost). For our r16 fleet that is `enp81s0f1` (10.194.134.0/22).

---

## Parallelism cheat sheet

| Mode | Spans nodes? | Needs Ray? | Example |
|------|--------------|------------|---------|
| **TP** (tensor parallel) | Yes, if TP > GPUs/node | **Yes** | 64 GPUs, 8 nodes → TP=8, PP=8 |
| **PP** (pipeline parallel) | Yes | **Yes** | Pipeline stages on different nodes |
| **DP** (data parallel replicas) | Can | Often **no** | Independent copies; nginx or vLLM native DP |
| **Single-node TP=8** | No | **No** | Kimi-K3 on one MI355X |

Common multi-node pattern ([vLLM docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)):

- `tensor_parallel_size` = GPUs **per node**
- `pipeline_parallel_size` = **number of nodes**

Alternative: `tensor_parallel_size` = total cluster GPUs (needs very fast interconnect).

---

## End-to-end flow (Slurm + Ray + vLLM)

```
1. sbatch          → Slurm reserves N nodes (ONE job)
2. HEAD = first node in SLURM_NODELIST
3. srun on head    → ray start --head --node-ip-address=<head_ip>
4. srun on workers → ray start --address=<head_ip>:6379
5. Wait            → ray status shows N×8 GPUs
6. On head only    → vllm serve … --distributed-executor-backend ray
7. Clients         → curl head:8000/v1/…
```

Slurm job skeleton:

```bash
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
```

**Do not** start Ray in one Slurm job and vLLM in another — both will fight for GPUs.

---

## Kimi-K3 8-node run (TP=8 × PP=8, 64 GPUs)

How the multi-node Kimi-K3 job was run: one model sharded across **8 nodes × 8 GPUs**
using Ray, submitted via Slurm on `sshetkud@smc200x-ccs-e12-31.cs-aus.dcgpu`.

### Topology & key facts

- **Nodes:** `smci355-ccs-aus-r16-[06,10,14,18,22,26,30,34]` (partition `mi355x-r16`, `--exclusive`).
- **Model:** `/data/afde/model/moonshotai/Kimi-K3` (node-local on all 8) → `/model_weights` in container.
- **Image:** `vllm/vllm-openai-rocm:kimi-k3`. **Note:** this image ships **without Ray**; the
  launcher `pip install ray[default]` (v2.57.0, ~6–7 min) into each container at startup.
- **Ray control plane:** interface `enp81s0f1` (10.194.134.0/22). Head = first node (r16-06);
  workers join `HEAD_IP:6379`. Head IP is shared via an NFS file; workers block until the head
  signals `DONE`.
- **Collective comms:** `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME=enp81s0f1`; RCCL uses the
  Pensando `benic*` RDMA fabric for tensor traffic.
- **Gotchas hit & fixed:**
  1. Image lacks Ray → pip-install in launcher.
  2. Stale VRAM from leftover containers fails vLLM's startup memory check →
     `docker rm -f` / free VRAM on all nodes before submit.

### How to run

```bash
# from the controller, in the bench dir where both scripts live:
cd /mnt/dcgpuval/afde/sshetkud/kimi-k3-bench
sbatch run_kimi_k3_vllm_8n_ray.sbatch
# watch bring-up:
tail -f /mnt/dcgpuval/afde/sshetkud/kimi-k3-vllm-8n-<JID>.log
```

### Pre-submit cleanup (avoid stale-VRAM failures)

vLLM aborts at startup if a GPU isn't (nearly) empty
(`Free memory on device … less than desired GPU memory utilization`).
Clear leftover project containers / VRAM on all nodes first:

```bash
srun -p mi355x-r16 --nodelist=smci355-ccs-aus-r16-[06,10,14,18,22,26,30,34] \
  -N8 --ntasks-per-node=1 -t 00:05:00 bash -lc '
    sudo docker rm -f kimi_k3_ray kimi_k3 sglang_job atom_job 2>/dev/null || true
    sudo pkill -9 -f "vllm|ray::|raylet|gcs_server|EngineCore" 2>/dev/null || true'
```

---

## `run_kimi_k3_vllm_8n_ray.sbatch`

```bash
#!/bin/bash
#SBATCH -J kimi-k3-vllm-8n-ray
#SBATCH -p mi355x-r16
#SBATCH -N 8
#SBATCH --nodelist=smci355-ccs-aus-r16-[06,10,14,18,22,26,30,34]
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH -t 16:00:00
#SBATCH -o /mnt/dcgpuval/afde/sshetkud/kimi-k3-vllm-8n-%j.log
#SBATCH -e /mnt/dcgpuval/afde/sshetkud/kimi-k3-vllm-8n-%j.log

set -uo pipefail

export JOB_ID="${SLURM_JOB_ID}"
export IMG=vllm/vllm-openai-rocm:kimi-k3
export MODEL_HOST=/data/afde/model/moonshotai/Kimi-K3
export MODEL_IN=/model_weights
export CNAME=kimi_k3_ray
export PORT=8000
export RESULT_DIR=/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench
export RAY_IFACE=enp81s0f1
export GPUS_PER_NODE=8
export TP=8
export PP=8
export MAX_MODEL_LEN=10240

mkdir -p "$RESULT_DIR"
rm -f "${RESULT_DIR}/ray_head_${JOB_ID}.ip" "${RESULT_DIR}/ray_done_${JOB_ID}.flag"
rm -rf "${RESULT_DIR}/ray_ready_${JOB_ID}"

HEAD_HOST="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
echo "[$(date -Is)] JOB=$JOB_ID nodes=$SLURM_JOB_NODELIST head=$HEAD_HOST TPxPP=${TP}x${PP}"

LAUNCHER="${RESULT_DIR}/ray_launcher.sh"

# One launcher per node; step stays alive until the head signals DONE (workers block on it).
srun --ntasks=8 --ntasks-per-node=1 --gpus-per-node=8 \
  bash "$LAUNCHER" "$HEAD_HOST"

echo "[$(date -Is)] all srun tasks returned; job complete. results=${RESULT_DIR}/vllm8n_bench_${JOB_ID}"
```

---

## `ray_launcher.sh`

Runs once per node (`srun --ntasks-per-node=1`). Decides head vs worker, installs Ray, forms the
cluster, and on the head launches `vllm serve` + the benchmark sweep. Workers block until `DONE`.

```bash
#!/usr/bin/env bash
# Ray + vLLM multi-node launcher. Runs once per node via `srun --ntasks-per-node=1`.
# NOTE: the vllm/vllm-openai-rocm:kimi-k3 image ships WITHOUT ray, so we pip-install
#       ray[default] into each container (parallel across nodes, ~6-7 min) before use.
set -uo pipefail

: "${IMG:?}" "${MODEL_HOST:?}" "${MODEL_IN:?}" "${CNAME:?}" "${PORT:?}"
: "${RESULT_DIR:?}" "${JOB_ID:?}" "${RAY_IFACE:?}" "${TP:?}" "${PP:?}"
: "${MAX_MODEL_LEN:?}" "${GPUS_PER_NODE:?}"

HEAD_HOST="$1"
ME="$(hostname -s)"
HEAD_IP_FILE="${RESULT_DIR}/ray_head_${JOB_ID}.ip"
DONE_FILE="${RESULT_DIR}/ray_done_${JOB_ID}.flag"
READY_DIR="${RESULT_DIR}/ray_ready_${JOB_ID}"
mkdir -p "$READY_DIR"

log(){ echo "[$(date -Is)] [$ME] $*"; }
fail(){ log "ERROR: $*"; touch "$DONE_FILE" 2>/dev/null || true; exit 1; }

MY_IP="$(ip -o -4 addr show "$RAY_IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
[ -n "$MY_IP" ] || fail "no IPv4 on $RAY_IFACE"
log "control-plane ip on $RAY_IFACE = $MY_IP"

DOCKER_COMMON=(--rm -d --name "$CNAME"
  --device=/dev/kfd --device=/dev/dri --group-add video
  --shm-size 64G --network host --ipc host
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE
  -v "${MODEL_HOST}:${MODEL_IN}:ro"
  -v "${RESULT_DIR}:${RESULT_DIR}"
  --env VLLM_ROCM_USE_AITER=1
  --env VLLM_HOST_IP="$MY_IP"
  --env NCCL_SOCKET_IFNAME="$RAY_IFACE"
  --env GLOO_SOCKET_IFNAME="$RAY_IFACE"
  --env RAY_DEDUP_LOGS=0)

cleanup(){ sudo docker rm -f "$CNAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

sudo docker rm -f "$CNAME" >/dev/null 2>&1 || true
log "docker pull $IMG"; sudo docker pull "$IMG" >/dev/null || fail "docker pull failed"

sudo docker run "${DOCKER_COMMON[@]}" --entrypoint bash "$IMG" -lc 'sleep infinity' || fail "docker run failed"
sleep 3

# ---- install ray into this container (image lacks it) ----
log "installing ray[default] into container (~6-7 min)..."
sudo docker exec "$CNAME" bash -lc "pip install -q 'ray[default]'" || fail "ray pip install failed"
RV=$(sudo docker exec "$CNAME" bash -lc "ray --version 2>/dev/null")
log "ray installed: ${RV:-unknown}"

if [ "$ME" = "$HEAD_HOST" ]; then
  # ---------------- HEAD ----------------
  log "starting Ray HEAD at ${MY_IP}:6379"
  sudo docker exec "$CNAME" bash -lc "ray start --head --node-ip-address=${MY_IP} --port=6379 --num-gpus=${GPUS_PER_NODE} --dashboard-host=0.0.0.0 --disable-usage-stats" || fail "ray head start failed"
  echo "$MY_IP" > "$HEAD_IP_FILE"

  WANT=$(( TP * PP ))
  log "waiting for ${WANT} GPUs to join the Ray cluster..."
  for i in $(seq 1 180); do
    GOT=$(sudo docker exec "$CNAME" bash -lc "ray status 2>/dev/null | grep -oE '[0-9]+\.[0-9]+/[0-9]+\.[0-9]+ GPU' | head -1 | sed -E 's#.*/##; s/(\.[0-9]+)? GPU//'" 2>/dev/null)
    GOT_INT=${GOT%%.*}
    [ -n "$GOT_INT" ] && log "ray GPUs so far: ${GOT_INT}/${WANT}"
    [ "${GOT_INT:-0}" = "$WANT" ] && { log "all ${WANT} GPUs present"; break; }
    sleep 10
  done

  log "launching vLLM serve (TP=${TP} PP=${PP}, total $((TP*PP)) GPUs)"
  sudo docker exec "$CNAME" bash -lc "
    vllm serve ${MODEL_IN} --served-model-name kimi-k3 \
      --dtype auto --trust-remote-code \
      --tensor-parallel-size ${TP} --pipeline-parallel-size ${PP} \
      --distributed-executor-backend ray \
      --gpu-memory-utilization 0.90 --no-enable-prefix-caching \
      --max-model-len ${MAX_MODEL_LEN} --max-num-seqs 256 \
      --reasoning-parser kimi_k3 --language-model-only \
      --disable-uvicorn-access-log --host 0.0.0.0 --port ${PORT} \
      > ${RESULT_DIR}/vllm_serve_${JOB_ID}.log 2>&1 &
  "

  log "waiting for /health (up to 300 min)"
  READY=0
  for i in $(seq 1 600); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then READY=1; log "server ready after ${i} probes"; break; fi
    sleep 30
  done
  [ "$READY" = "1" ] || { sudo docker exec "$CNAME" bash -lc "tail -n 100 ${RESULT_DIR}/vllm_serve_${JOB_ID}.log" || true; fail "server never became ready"; }

  OUTDIR="${RESULT_DIR}/vllm8n_bench_${JOB_ID}"; mkdir -p "$OUTDIR"
  for c in 16 32 64 128 256; do
    NP=$(( c * 10 ))
    log "benchmark concurrency=${c} num_prompts=${NP}"
    sudo docker exec "$CNAME" bash -lc "
      cd /tmp && vllm bench serve --model ${MODEL_IN} --served-model-name kimi-k3 \
        --base-url http://localhost:${PORT} \
        --percentile-metrics ttft,tpot,itl,e2el \
        --dataset-name random --ignore-eos --temperature 0 \
        --trust-remote-code --max-concurrency ${c} \
        --num-prompts ${NP} --random-input-len 8192 --random-output-len 1024 \
        --random-range-ratio 0 --save-result \
        --result-filename ${OUTDIR}/kimi_k3_vllm8n_c${c}_i8192_o1024.json
    " 2>&1 | tee "${OUTDIR}/bench_c${c}.log"
    grep -E 'Total Token throughput|Output token throughput|Mean TTFT' "${OUTDIR}/bench_c${c}.log" || true
  done

  log "DONE — results in ${OUTDIR}"
  touch "$DONE_FILE"

else
  # ---------------- WORKER ----------------
  log "waiting for head ip file ${HEAD_IP_FILE}"
  for i in $(seq 1 300); do [ -s "$HEAD_IP_FILE" ] && break; sleep 5; done
  HEAD_IP="$(cat "$HEAD_IP_FILE" 2>/dev/null)"
  [ -n "$HEAD_IP" ] || fail "no head ip after wait"
  log "joining Ray head at ${HEAD_IP}:6379"
  sudo docker exec "$CNAME" bash -lc "ray start --address=${HEAD_IP}:6379 --num-gpus=${GPUS_PER_NODE} --disable-usage-stats" || fail "ray worker join failed"
  touch "${READY_DIR}/${ME}.joined"
  log "worker joined; blocking until head signals DONE"
  while [ ! -f "$DONE_FILE" ]; do sleep 30; done
  log "DONE flag seen; worker exiting"
fi
```

---

## Lab patterns compared

| Pattern | Nodes | Ray? | How it scales |
|---------|-------|------|----------------|
| **Kimi-K3 single-node** (Grafana submit) | 1 | No | TP=8, `mp` backend |
| **Kimi-K3 8n Ray** (this doc) | 8 | **Yes** | TP=8 × PP=8, one engine |
| **vLLM-job replicas** | N | No | N independent engines + nginx LB |
| **vLLM-job native DP** | N | No | Cross-node DP rendezvous |

For models that **fit on one MI355X node**, replicas + load balancer usually beats
cross-node TP/PP over TCP for latency-sensitive serving.

---

## Networking & ops checklist

- [ ] Head/worker IP on **fabric NIC** (`enp81s0f1` on r16), not management-only
- [ ] Ports **6379** (Ray GCS) reachable node-to-node
- [ ] `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` set consistently
- [ ] Model weights visible on every node (node-local or NFS)
- [ ] Stale containers cleared before submit
- [ ] **One Slurm job** owns all nodes for the full Ray + vLLM lifetime
- [ ] Cleanup: `ray stop` + `docker rm` when job ends

Debug: Ray dashboard `:8265`, `ray status` on head, `vllm_serve_<JID>.log` on NFS.

---

## References

- [vLLM Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [vLLM forum: multi-node DP with Slurm](https://discuss.vllm.ai/t/running-vllm-multi-node-data-parallel-with-slurm/1362)
- [MeluXina: multi-node vLLM + Slurm + Ray](https://docs.lxp.lu/howto/llama3-vllm/)
