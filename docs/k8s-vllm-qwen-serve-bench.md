# vLLM serve + benchmark on Kubernetes (Qwen3.5-35B-A3B, 8x MI300X)

End-to-end runbook for serving **Qwen3.5-35B-A3B** with vLLM on a **reserved**
Kubernetes node (`dell-ccs-e14-34`, 8x MI300X / gfx942) and running a
`vllm bench serve` concurrency sweep against it — plus every issue hit along the
way and how it was fixed.

Related docs: [k8s-node-reservation.md](k8s-node-reservation.md),
[vllm-server.md](vllm-server.md), [vllm-benchmark.md](vllm-benchmark.md).

Manifests: [`manifests/vllm-qwen-deploy.yaml`](manifests/vllm-qwen-deploy.yaml),
[`manifests/vllm-bench-qwen.yaml`](manifests/vllm-bench-qwen.yaml).

---

## Environment

| Item | Value |
|---|---|
| Node | `dell-ccs-e14-34` — 8x AMD MI300X (gfx942) |
| Reservation | taint+label `reservation.lab.amd/owner=sshetkud` (see reservation doc) |
| Image | `vllm/vllm-openai-rocm:kimi-k3` |
| Model | `Qwen3.5-35B-A3B` (BF16 MoE, ~3B active), TP=8 |
| Model store | NFS `/mnt/dcgpuval` (model dir is a symlink → mount the whole root) |
| Access | Windows workstation → `portal` jump host → `kubectl` / node shell |

`kubectl` is driven from the workstation through the `portal` jump host. Direct
node shell (for `docker`, disk checks) uses a ProxyCommand:

```bash
ssh -o ProxyCommand="ssh -W %h:%p -o ClearAllForwardings=yes -o ControlMaster=no -o ControlPath=none portal" \
    -l sshetkud -i C:/Users/sshetkud/.ssh/id_ed25519 dell-ccs-e14-34
```

---

## 1. Serve

Apply the Deployment + Service ([`manifests/vllm-qwen-deploy.yaml`](manifests/vllm-qwen-deploy.yaml)):

```bash
kubectl apply -f manifests/vllm-qwen-deploy.yaml
kubectl rollout status deploy/vllm-qwen
kubectl logs -f deploy/vllm-qwen        # wait for "Application startup complete"
```

Key points of the serve manifest:

- **Pin to the reserved node** with both `nodeSelector` **and** a matching
  `toleration` for `reservation.lab.amd/owner=sshetkud:NoSchedule`, plus a
  `hostname In [dell-ccs-e14-34]` affinity.
- `command: ["vllm","serve","/mnt/dcgpuval/models/Qwen/Qwen3.5-35B-A3B"]` with
  `--served-model-name Qwen3.5-35B-A3B -tp 8 --max-model-len 10240
  --gpu-memory-utilization 0.95 --max-num-seqs 128 --trust-remote-code`.
- `resources.limits."amd.com/gpu": "8"` claims all 8 GPUs.
- `imagePullPolicy: IfNotPresent` — reuse the on-node image (see issue #5).
- Model volume is a **hostPath to the NFS root** `/mnt/dcgpuval` (type `Directory`)
  because the model path is a symlink into that root. A 64Gi `/dev/shm` emptyDir
  (Memory) is added for NCCL/RCCL shared memory.
- A ClusterIP `Service` named `vllm-qwen` exposes port 8000 in-cluster, which the
  benchmark Job targets at `http://vllm-qwen:8000`.

---

## 2. Benchmark

Apply the Job ([`manifests/vllm-bench-qwen.yaml`](manifests/vllm-bench-qwen.yaml)):

```bash
kubectl apply -f manifests/vllm-bench-qwen.yaml
kubectl logs -f job/vllm-bench-qwen
```

What it does:

- Waits for `GET /v1/models` to return `200` on the in-cluster Service.
- Sweeps `--max-concurrency` over `16, 32, 64, 128` (num-prompts = 5x concurrency,
  min 64), random dataset **8192 in / 1024 out**, `--ignore-eos`.
- Critically passes **`--tokenizer /mnt/dcgpuval/models/Qwen/Qwen3.5-35B-A3B`** (a
  local path) while keeping `--model Qwen3.5-35B-A3B` for the API payload — this
  stops `vllm bench serve` from trying to pull the tokenizer from HuggingFace for
  a locally-served model (which returned HTTP 401, issue #8).
- Saves per-concurrency JSON to `/mnt/dcgpuval/bench_qwen_<timestamp>/`.

---

## 3. Results

Concurrency sweep, ISL 8192 / OSL 1024, `dell-ccs-e14-34` (8x MI300X, TP=8),
results in `/mnt/dcgpuval/bench_qwen_20260929_005827/`, **0 failed requests**:

| Concurrency | Total throughput (tok/s) |
|---|---|
| 16  | 4,173 |
| 32  | 7,936 |
| 64  | 13,279 |
| 128 | 22,188 |

Throughput scales cleanly to the peak **22,188 tok/s @ concurrency 128**.

---

## 4. Issues seen and how they were fixed

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | **Eviction storm** — hundreds of `vllm-qwen` pods `Evicted` on the node | pods tolerated `disk-pressure`, so they kept landing on a node under real disk pressure and the kubelet eviction manager kept evicting them | scale deploy to 0, **remove the disk-pressure toleration**, delete Failed pods, free disk |
| 2 | `ssh portal` failed: `getsockname failed: Bad file descriptor` | `portal` config has `LocalForward 8443` + `ExitOnForwardFailure yes` | add `-o ClearAllForwardings=yes -o ControlMaster=no -o ControlPath=none` |
| 3 | `portal → node` hop: `Permission denied (publickey)` | no agent forwarding; wrong default username | ProxyCommand with explicit `-l sshetkud -i C:/Users/sshetkud/.ssh/id_ed25519` (see Environment) |
| 4 | Node `DiskPressure=True`, scheduling blocked | root FS 91% full | delete stale `~/.cache/huggingface` (68G) → 75% used |
| 5 | Image **re-pulled** despite `IfNotPresent` | kubelet **image garbage collection** deleted the vLLM image while the node was under disk pressure | expected one-time behavior; the real fix is to keep free space above the GC threshold (issue #6) |
| 6 | Pull killed mid-way / DiskPressure re-triggered / image rolled back — an **infinite pull loop** | the large ROCm image needs ~2x space transiently (compressed layers + extracted snapshots, ~60-80G peak); this dipped free space below the 15% threshold → GC nuked the in-progress image → loop | `docker image prune -a -f` freed **85.13GB** (removed unused `rocm/jax-training:maxtext-v26.3` and `vllm/vllm-openai-rocm:v0.19.1`) → ~188G free buffer; pull then completed (~40 min at ~0.5GB/min egress) |
| 7 | `DiskPressure` slow to clear even after freeing space | kubelet's 5-minute **pressure-transition period** before flipping the condition back to `False` | wait out the transition window; verify taint `node.kubernetes.io/disk-pressure` is gone |
| 8 | Benchmark **HF 401**: `Repository Not Found ... huggingface.co/api/models/Qwen3.5-35B-A3B` | `vllm bench serve` tried to fetch the tokenizer from HF for a locally-served model | add `--tokenizer /mnt/dcgpuval/models/Qwen/Qwen3.5-35B-A3B` (keep `--model` for the API payload) |
| 9 | Jump host `kex_exchange_identification: Connection closed by remote host` | too many rapid SSH sessions → Conductor gateway per-IP throttle / fail2ban | wait ~2 min and reconnect; avoid session storms |

### PowerShell / remote-shell quoting pitfalls (recurring)

- `$(...)` is interpolated **locally** by PowerShell → wrap remote command strings
  in **single quotes**.
- `grep -iE "a|b"` alternation breaks through the Conductor login shell (the `|`
  splits the command → "command not found") → use single-word `grep -i disk`.
- jsonpath with `()` / `\` escapes breaks → **base64-encode the YAML locally** and
  pipe `base64 -d | kubectl apply -f -`.
- Trim noisy output with `Select-Object -Last N` / `Select-String`.

---

## TL;DR

The whole exercise came down to **disk hygiene on the node** (a large ROCm image
plus a low free-space margin caused a GC/eviction/re-pull loop) and **one
benchmark flag** (`--tokenizer <local path>`). Once ~150G was freed and the
tokenizer was pointed at the local model, serve reached
`Application startup complete` and the sweep peaked at **22,188 tok/s @ c128** with
zero failures.
