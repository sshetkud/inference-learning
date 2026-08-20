# Multi-node vLLM on Ray (Kimi-K3, 8×MI355X, TP×PP)

How the multi-node vLLM Kimi-K3 job was run: a single model sharded across **8 nodes ×
8 GPUs = 64 GPUs** (tensor-parallel **TP=8** within a node, pipeline-parallel **PP=8**
across nodes) using a **Ray** cluster, submitted via Slurm on
`sshetkud@smc200x-ccs-e12-31.cs-aus.dcgpu`.

Log/result locations for this job are in [log-file-locations.md](log-file-locations.md).

---

## Topology & key facts

- **Nodes:** `smci355-ccs-aus-r16-[06,10,14,18,22,26,30,34]` (partition `mi355x-r16`, `--exclusive`).
- **Model:** `/data/afde/model/moonshotai/Kimi-K3` (node-local on all 8) → `/model_weights` in container.
- **Image:** `vllm/vllm-openai-rocm:kimi-k3`. **Note:** this image ships **without Ray**, so the
  launcher `pip install ray[default]` (v2.57.0, ~6.5 min) into each container at startup.
- **Ray control plane:** interface `enp81s0f1` (10.194.134.0/22). Head = first node (r16-06);
  workers join `HEAD_IP:6379`. Head IP is shared via an NFS file; workers block until the head
  signals `DONE`.
- **Collective comms:** `NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME=enp81s0f1`; RCCL uses the Pensando
  `benic*` RDMA fabric for tensor traffic.
- **Gotchas hit & fixed:** (1) image lacks Ray → pip-install in launcher; (2) stale VRAM from
  leftover containers on a node fails vLLM's startup free-memory check → clean stale
  `docker rm -f` / free VRAM on all nodes before submit.

## How to run

```bash
# from the controller, in the bench dir where both scripts live:
cd /mnt/dcgpuval/afde/sshetkud/kimi-k3-bench
sbatch run_kimi_k3_vllm_8n_ray.sbatch
# watch bring-up:
tail -f /mnt/dcgpuval/afde/sshetkud/kimi-k3-vllm-8n-<JID>.log
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

## Pre-submit cleanup (avoid stale-VRAM failures)

vLLM aborts at startup if a GPU isn't (nearly) empty
(`Free memory on device cuda:N ... less than desired GPU memory utilization`).
Clear leftover project containers / VRAM on all nodes first:

```bash
srun -p mi355x-r16 --nodelist=smci355-ccs-aus-r16-[06,10,14,18,22,26,30,34] \
  -N8 --ntasks-per-node=1 -t 00:05:00 bash -lc '
    sudo docker rm -f kimi_k3_ray kimi_k3 sglang_job atom_job 2>/dev/null || true
    sudo pkill -9 -f "vllm|ray::|raylet|gcs_server|EngineCore" 2>/dev/null || true'
```
