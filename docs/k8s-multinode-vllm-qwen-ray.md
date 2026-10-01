# Multi-node vLLM on Kubernetes via KubeRay (Qwen3.5-35B-A3B, 4× MI300X nodes / 32 GPU)

End-to-end runbook for serving **one** vLLM engine **sharded across 4 reserved
Kubernetes nodes** (`dell-ccs-e14-[22,28,34,40]`, 8× MI300X / gfx942 each = **32 GPU**)
using a **KubeRay `RayCluster`** — `TP=8` intra-node × `PP=4` across nodes — then running
a `vllm bench serve` concurrency sweep against it. Plus every issue hit and how it was fixed.

This is the **Kubernetes/KubeRay** companion to:

- [k8s-vllm-qwen-serve-bench.md](k8s-vllm-qwen-serve-bench.md) — the **single-node** (1× node, TP=8, plain Deployment) version of this same model.
- [multinode-vllm-ray.md](multinode-vllm-ray.md) — the **Slurm**-based Ray concepts + the Kimi-K3 8-node (TP×PP) runbook.

Related: [k8s-node-reservation.md](k8s-node-reservation.md), [vllm-server.md](vllm-server.md), [vllm-benchmark.md](vllm-benchmark.md).

Manifests: [`manifests/raycluster-vllm-qwen-mn.yaml`](manifests/raycluster-vllm-qwen-mn.yaml),
[`manifests/vllm-qwen-mn-svc.yaml`](manifests/vllm-qwen-mn-svc.yaml),
[`manifests/vllm-bench-qwen-mn.yaml`](manifests/vllm-bench-qwen-mn.yaml).

---

## Single-node vs multi-node — why Ray at all?

The single-node run serves Qwen3.5-35B-A3B with a plain `Deployment` (`TP=8`, vLLM's own
`mp` executor) and peaks at **22,188 tok/s @ c128** — because the model *fits on one node*.

This doc does the opposite on purpose: force **one logical engine to span 4 nodes** so the
distributed path (KubeRay + Ray executor + cross-node pipeline parallelism) is exercised
on Kubernetes. It is the setup you need when a model is **too big for one node**; for a
model that already fits, 4 independent replicas behind a load balancer will beat cross-node
TP/PP for throughput (see [Results](#4-results--read)).

```
RayCluster vllm-qwen-mn                        (Ray 2.51.1, KubeRay operator)
 ├─ head   → 1 pod, 8 GPU   (also runs `vllm serve`)   on dell-ccs-e14-22
 └─ worker → 3 pods, 8 GPU each                         on e14-28 / e14-34 / e14-40
      one pod per reserved node (nodeSelector + podAntiAffinity(hostname))
      → 32 GPU in a single Ray cluster

vllm serve Qwen3.5-35B-A3B  -tp 8  -pp 4  --distributed-executor-backend ray
        ↓  Service vllm-qwen-mn:8000  ←  Job vllm-bench-qwen-mn (c = 16/32/64/128)
```

---

## Environment

| Item | Value |
|---|---|
| Nodes | `dell-ccs-e14-[22,28,34,40]` — 4× (8× AMD MI300X / gfx942) = **32 GPU** |
| Reservation | taint+label `reservation.lab.amd/owner=sshetkud` (see reservation doc) |
| Orchestration | KubeRay operator, `ray.io/v1`, `rayVersion: 2.51.1` |
| Image | `local/vllm-ray-qwen:kimi-k3` (base `vllm/vllm-openai-rocm:kimi-k3` + `pip install ray[default]`, built/loaded on all 4 nodes) |
| Model | `Qwen3.5-35B-A3B` (BF16 MoE, 14 shards, ~68 GB), `TP=8 × PP=4` |
| Model store | shared NFS **`/mnt/y_share`** (`10.235.26.220:/mnt/y_share`), hostPath-mounted |
| Pod network | **`hostNetwork: true`** (node IPs on `eno8303`) — not the Calico overlay (see issue #1) |
| Cross-node coll. | NCCL over **TCP** (`NCCL_IB_DISABLE=1`); intra-node TP over xGMI/P2P (see issue #2) |
| Access | Windows workstation → `portal` jump host → `kubectl` / node shell |

`kubectl` is driven from the workstation through the `portal` jump host (which has
`kubectl`); direct node shell (`docker`, disk checks) uses a ProxyCommand:

```bash
ssh -o ProxyCommand="ssh -W %h:%p -o ClearAllForwardings=yes -o ControlMaster=no -o ControlPath=none portal" \
    -l sshetkud -i C:/Users/sshetkud/.ssh/id_ed25519 dell-ccs-e14-40
```

---

## 0. Stage the model on shared NFS

The e14 nodes do **not** mount the portal's `/mnt/dcgpuval` NFS (that is where the
single-node run read the model). Their `/mnt/dcgpuval` is a local empty dir. The real
NFS shared across all four e14 nodes is **`/mnt/y_share`**, so the weights have to be
copied there once (visible to every pod via hostPath):

```bash
# server-to-server pull of the resolved HF snapshot (follow symlinks with tar -h)
ssh portal "tar -C /mnt/dcgpuval/models/Qwen -chf - Qwen3.5-35B-A3B" \
  | tar -C /mnt/y_share/models/Qwen -xf -          # ~68 GB, ~3.3 GB/min
```

Make the output parent writable for the benchmark pod (NFS `root_squash` maps the pod to
`nobody`), so it can create its results dir:

```bash
sudo chmod 777 /mnt/y_share/models
```

---

## 1. Create the Ray cluster

```bash
kubectl apply -f manifests/raycluster-vllm-qwen-mn.yaml
kubectl get raycluster vllm-qwen-mn -w        # wait: head + 3 workers Ready
kubectl get pods -l app=vllm-qwen-mn -o wide  # confirm exactly one pod per e14 node
```

Key points of the RayCluster manifest:

- **One head + 3 workers**, `num-gpus: "8"` each; `podAntiAffinity` on `app=vllm-qwen-mn`
  (topologyKey `hostname`) pins exactly **one pod per node** — required because
  `hostNetwork` pods share the node's port space.
- **`hostNetwork: true` + `dnsPolicy: ClusterFirstWithHostNet`** — Ray/vLLM bind to node
  IPs (`10.235.x` on `eno8303`). `VLLM_HOST_IP` comes from `status.hostIP`.
- Pinned to the reservation with `nodeSelector` + matching `toleration`.
- `resources.limits."amd.com/gpu": 8` per pod; `privileged` + `IPC_LOCK`;
  64Gi `/dev/shm` (Memory emptyDir) for RCCL/NCCL; hostPath `/mnt/y_share` for the model.
- Env: `NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=eno8303`, `GLOO_SOCKET_IFNAME=eno8303`,
  `RAY_EXPERIMENTAL_NOSET_{HIP,ROCR}_VISIBLE_DEVICES=1`, `HF_HUB_OFFLINE=1`.

Verify Ray sees all 32 GPUs before serving:

```bash
HEAD=$(kubectl get pod -l ray.io/cluster=vllm-qwen-mn,ray.io/node-type=head -o name)
kubectl exec -it "$HEAD" -- ray status        # expect 32.0 GPU across 4 nodes
```

---

## 2. Serve (exec vLLM into the head)

vLLM is **not** started by the manifest — the head container just runs `ray start`. Once
the cluster is Ready, launch `vllm serve` inside the head pod; vLLM then places 32 workers
across the Ray cluster:

```bash
kubectl exec -it "$HEAD" -- bash -lc '
  vllm serve /mnt/y_share/models/Qwen/Qwen3.5-35B-A3B \
    --served-model-name Qwen3.5-35B-A3B \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 4 \
    --distributed-executor-backend ray \
    --gpu-memory-utilization 0.95 \
    --max-model-len 10240 \
    --max-num-seqs 128 \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000
'
# wait for "Application startup complete"
```

Expose it and smoke-test:

```bash
kubectl apply -f manifests/vllm-qwen-mn-svc.yaml      # ClusterIP vllm-qwen-mn:8000 → head pod
kubectl get endpoints vllm-qwen-mn                    # should be the head node IP :8000
kubectl exec -it "$HEAD" -- curl -s http://localhost:8000/v1/models
```

The smoke test returned the model (`root /mnt/y_share/models/Qwen/Qwen3.5-35B-A3B`,
`max_model_len 10240`) and a completion with
`system_fingerprint: ...-tp8-pp4-...` — confirming one engine over 32 GPUs.

---

## 3. Benchmark

```bash
kubectl apply -f manifests/vllm-bench-qwen-mn.yaml
kubectl logs -f job/vllm-bench-qwen-mn
# per-concurrency JSON lands in /mnt/y_share/models/bench_qwen_mn_<timestamp>/c*.json
```

The Job (CPU-only client pod):

- Waits for `GET /v1/models` → `200` on the in-cluster Service `http://vllm-qwen-mn:8000`.
- Sweeps `--max-concurrency` over `16, 32, 64, 128` (num-prompts = 5× concurrency, min 64),
  random dataset **8192 in / 1024 out**, `--ignore-eos`.
- Passes **`--tokenizer /mnt/y_share/models/Qwen/Qwen3.5-35B-A3B`** (local path) while
  keeping `--model Qwen3.5-35B-A3B` for the API payload — stops `vllm bench serve` from
  pulling the tokenizer from HuggingFace (HTTP 401 for a locally-served model).

---

## 4. Results + read

Concurrency sweep, ISL 8192 / OSL 1024, one engine over 4× MI300X (TP=8 × PP=4),
**0 failed requests** at every level:

| Conc | Prompts | Dur (s) | req/s | Output tok/s | **Total tok/s** | TTFT med (ms) | TTFT p99 (ms) | TPOT med (ms) | E2EL med (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16  | 80  | 340.6  | 0.235 | 240.5 | **2,164** | 13,197 | 34,566  | 52.96  | 67,254  |
| 32  | 160 | 509.0  | 0.314 | 321.9 | **2,897** | 13,212 | 60,705  | 86.21  | 101,371 |
| 64  | 320 | 853.2  | 0.375 | 384.1 | **3,457** | 13,175 | 119,037 | 152.99 | 169,667 |
| 128 | 640 | 1520.9 | 0.421 | 430.9 | **3,878** | 13,084 | 233,550 | 282.41 | 302,204 |

**How to read it:**

- **Throughput scales sub-linearly:** 8× the concurrency buys only ~1.8× total tok/s
  (2,164 → 3,878). The engine saturates early.
- **TTFT median is flat (~13 s)** across all loads — dominated by the 8192-token prefill,
  not queueing. But **TTFT p99 explodes** (35 s → 234 s) as the queue backs up under load.
- **TPOT (decode) degrades ~linearly** with load (53 → 282 ms/token).
- This is the classic **cross-node PP over TCP** signature. Compare to the single-node run
  (same model, TP=8, `mp`): **22,188 tok/s @ c128** vs **3,878 tok/s** here. For a model
  that *fits on one node*, cross-node PP is a big throughput tax — use it only when the
  model does not fit, otherwise run **N single-node replicas behind a load balancer**.

---

## 5. Issues seen and how they were fixed

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Engine init `No route to host` to the e14-28 pod (`10.244.x:100xx`); restarting `calico-node` made it flakier | the **Calico pod overlay** was unreliable across these e14 nodes | switch all Ray pods to **`hostNetwork: true` + `ClusterFirstWithHostNet`** so they use node IPs on `eno8303` (same path as NFS/SSH); add `podAntiAffinity` for one-pod-per-node to avoid host-port collisions; set `VLLM_HOST_IP` from `status.hostIP` |
| 2 | NCCL **`internal error` / `socketFinalizeAccept ... wrong type 3 != 4`** at engine init | under hostNetwork the **RoCE (mlx5) OOB handshake** got crossed between ranks | **`NCCL_IB_DISABLE=1`** → TCP transport over `eno8303` for cross-node PP (activations only, low bandwidth); intra-node TP still uses xGMI/P2P and is unaffected |
| 3 | Model path empty inside the pod (`/mnt/dcgpuval` was empty) | e14 nodes don't mount the portal NFS; their `/mnt/dcgpuval` is a local empty dir | stage weights to the **real shared NFS `/mnt/y_share`** and hostPath-mount that (see step 0) |
| 4 | Server-to-server `tar | tar` copy failed `Permission denied (publickey)` both directions | Conductor validates SSH keys against its own registry (`AuthorizedKeysCommand`); throwaway keys don't authenticate | run the pull with the user's **Conductor-registered** key (briefly placed on the node, `shred -u` immediately after) |
| 5 | Bench pod could not create its results dir on NFS | NFS `root_squash` maps the pod's root to `nobody` | `sudo chmod 777 /mnt/y_share/models` on a node before launching the Job |
| 6 | One worker in **CrashLoopBackOff** (raylet healthz liveness probe failing) | transient raylet start on a busy node | `kubectl delete pod` → rescheduled healthy on the now-idle node |
| 7 | e14-28 repeated **ImagePullBackOff** | kubelet **image GC** deleted the image above the disk high-threshold | prune node disk (77% → 26%) so GC stops evicting the image; reload image |
| 8 | Bench **HF 401** pulling the tokenizer | `vllm bench serve` tried HF for a locally-served model | add `--tokenizer <local path>` (keep `--model` for the API payload) |
| 9 | Jump-host SSH: `kex_exchange_identification` / `banner exchange` timeouts / publickey denied | rapid SSH sessions → Conductor per-IP throttle | wait ~45–60 s and reconnect; avoid session storms |

### PowerShell / remote-shell quoting pitfalls (recurring)

- `&&`, escaped `\"..\"`, `for` loops and redirections mangle through the PowerShell →
  `ssh portal "..."` path. **Reliable pattern:** write a bash script to a local file, then
  either pipe it over stdin (`Get-Content script -Raw | ssh ... "kubectl exec -i POD -- bash -s"`)
  or `scp` it to the host, `sed -i 's/\r$//'`, and run it.
- `scp` to compute nodes needs an explicit `sshetkud@host:` (defaults to the wrong `amd\sshetkud`).

---

## 6. Teardown & rebuild from scratch

The vLLM engine runs **inside the Ray head pod**, so deleting the RayCluster
deletes all head+worker pods, kills the engine, and frees the 32 GPUs in one step
— there is no separate "stop vLLM" action.

### Teardown

```bash
kubectl delete raycluster/vllm-qwen-mn svc/vllm-qwen-mn job/vllm-qwen-mn-bench --ignore-not-found
# verify everything is gone (both should report NotFound / no resources):
kubectl get raycluster vllm-qwen-mn
kubectl get pods -l app=vllm-qwen-mn
```

Deleting the `RayCluster` CR is what the KubeRay operator watches — it garbage-collects
the head and worker pods. Delete the `Service` and bench `Job` explicitly (they are not
owned by the RayCluster). Wait until all pods are gone before recreating, so the
`podAntiAffinity` (one pod per node) can place the new pods.

### Rebuild from scratch

```bash
# 1. recreate the cluster + service
kubectl apply -f manifests/raycluster-vllm-qwen-mn.yaml
kubectl get raycluster vllm-qwen-mn -w          # wait STATUS=ready
kubectl get pods -l app=vllm-qwen-mn -o wide    # 4/4 Running, one per e14 node

# 2. confirm all 32 GPUs joined the Ray cluster
HEAD=$(kubectl get pod -l ray.io/cluster=vllm-qwen-mn,ray.io/node-type=head -o name)
kubectl exec -it "$HEAD" -c ray-head -- ray status    # expect 32.0 GPU

# 3. launch the distributed engine inside the head (detached)
kubectl exec "$HEAD" -c ray-head -- bash -lc 'setsid bash -c "\
  vllm serve /mnt/y_share/models/Qwen/Qwen3.5-35B-A3B \
    --served-model-name Qwen3.5-35B-A3B -tp 8 -pp 4 --distributed-executor-backend ray \
    --gpu-memory-utilization 0.95 --max-model-len 10240 --max-num-seqs 128 --trust-remote-code \
    --host 0.0.0.0 --port 8000 > /tmp/vllm_serve.log 2>&1" </dev/null & echo launched'

# 4. poll startup, then smoke test
kubectl exec "$HEAD" -c ray-head -- tail -n 20 /tmp/vllm_serve.log   # wait "init engine ... took"
kubectl exec "$HEAD" -c ray-head -- curl -s http://localhost:8000/v1/models
```

Expected timings on this fleet: pods Ready ~1 min; weight load from the
`/mnt/y_share` NFS ~5 min; engine init/warmup ~2 min → **~7-8 min** before
`/v1/models` answers. Two normal quirks when launching the detached engine over
`kubectl exec`: the exec call may return after ~60 s while the engine keeps
starting (the `setsid ... &` is detached — poll the log, don't re-launch), and
re-running the serve while it is already up is harmless (bind to :8000 fails fast).

> The [`k8s-multinode-vllm` MCP server](https://github.com/sshetkud/inference-learning)
> wraps this exact teardown/rebuild loop as `vllm_teardown(confirm=true)` →
> `raycluster_create(dry_run=false)` → `ray_status()` → `vllm_serve(start=true)` →
> `vllm_models()`.

## TL;DR

Serving one Qwen3.5-35B-A3B engine across **4 nodes / 32 MI300X** on Kubernetes came down
to **three fixes**: (1) drop the flaky Calico overlay for **`hostNetwork`** (node IPs),
(2) **`NCCL_IB_DISABLE=1`** so cross-node PP uses TCP instead of the crossed RoCE handshake,
and (3) stage the model on the **`/mnt/y_share`** NFS the e14 nodes actually mount. The
engine served cleanly and the sweep peaked at **3,878 tok/s @ c128 with zero failures** —
~5.7× *lower* than the single-node TP=8 run (22,188 tok/s), which is the expected cost of
cross-node pipeline parallelism over TCP for a model that already fits on one node. Reach
for this topology when the model **doesn't** fit; otherwise use single-node replicas + a load balancer.
