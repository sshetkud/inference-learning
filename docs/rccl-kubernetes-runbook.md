# Multi-node RCCL on Kubernetes (MI355X + Pensando ionic)

Runbook for validating **multi-node RDMA** from inside Kubernetes pods on AMD Instinct
MI355X nodes with Pensando `ionic` NICs — the prerequisite step before deploying
multi-node vLLM on a cluster.

The headline result: **`hostNetwork` plus a `/dev/infiniband` hostPath is enough**.
No Multus, no SR-IOV device plugin, no RDMA device plugin.

Related docs: [multinode-vllm-ray.md](multinode-vllm-ray.md),
[mori-moe-collective.md](mori-moe-collective.md),
[log-file-locations.md](log-file-locations.md).

---

## Why this test comes first

Under Slurm, containers run on the host network and RCCL sees the RDMA NICs directly.
Under Kubernetes the default is a CNI veth (Calico here), which RCCL cannot use for
RoCE — it silently falls back to TCP and multi-node bandwidth collapses.

That makes "can a pod drive RoCE at line rate?" the single highest-risk question in a
Slurm-to-Kubernetes migration. Answer it with `all_reduce_perf` *before* trying to debug
it underneath a failing vLLM launch.

## Result summary

2 nodes × 8 MI355X = 16 ranks, `all_reduce_perf`, native ionic path:

| Metric | Kubernetes | Slurm baseline | Ratio |
|---|---|---|---|
| Peak busbw | **200.4 GB/s** @ 1 GB | ~280 GB/s @ 1 GB | 72% |
| Avg busbw | **60.4 GB/s** | 82.4 GB/s | 73% |
| `#wrong` | 0 | 0 | ✅ |

Functionally correct, but leaving ~28% on the table. See
[Tuning the 200 GB/s ceiling](#tuning-the-200-gbs-ceiling).

## Environment

| Component | Value |
|---|---|
| Kubernetes | v1.31.14 (kubeadm), Calico CNI |
| Nodes | Ubuntu 24.04, kernel 6.8.0-111, ROCm 7.2.2 |
| GPUs | 8× MI355X (gfx950) per node, via `amdgpu-device-plugin` as `amd.com/gpu` |
| RDMA | 9× Pensando `ionic` links, all ACTIVE |
| Mgmt NIC | `enp81s0f1` |
| Image | `rocm/roce-workload:ubuntu24_rocm-7.0.2_rccl-7.0.2_anp-v1.2.0_ainic-1.117.5-a-77` |

The image is self-contained and maps onto the usual `RCCL_PATH` layout with
`RCCL_PATH=/root`:

| Component | Path |
|---|---|
| Open MPI 4.1.6rc4 | `/root/ompi/install/bin/mpirun` |
| rccl-tests | `/root/rccl-tests/build/all_reduce_perf` |
| RCCL | `/root/rccl/build/release/librccl.so.1.0` |
| ANP plugin | `/root/amd-anp/build/librccl-anp.so` |
| UCX | `/root/ucx/install/lib` |

Note the ANP plugin ships **inside the image** even though the hosts have none. A
container can supply a plugin the host lacks.

---

## Step 1 — Confirm the image is on both nodes

Air-gapped clusters have no registry access, so `imagePullPolicy: IfNotPresent` only
works if the image is already in containerd. On **each** compute node:

```bash
sudo crictl images | grep roce-workload
```

## Step 2 — Drain the nodes in Slurm

If the nodes are in both Slurm and Kubernetes, nothing coordinates the two schedulers
and an `sbatch` can land on top of your pods. Requires Slurm operator rights:

```bash
sudo scontrol update NodeName=<head> State=DRAIN Reason="k8s-rccl-validation"
sudo scontrol update NodeName=<worker> State=DRAIN Reason="k8s-rccl-validation"
sinfo -h -n <head>,<worker> -o '%n %t %E'
```

Without `sudo` this fails with `slurm_update error: Invalid user id`.

## Step 3 — Shared SSH keypair Secret

`mpirun` launches remote ranks over SSH. The image entrypoint generates a keypair per
container, so two pods would not trust each other — inject one shared key instead:

```bash
rm -rf /tmp/rccl-ssh && mkdir -p /tmp/rccl-ssh
ssh-keygen -t rsa -b 4096 -N '' -f /tmp/rccl-ssh/id_rsa -q
cat > /tmp/rccl-ssh/config <<'CFG'
Host *
  Port 2222
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  LogLevel ERROR
CFG
kubectl create secret generic rccl-ssh --from-file=/tmp/rccl-ssh/
```

## Step 4 — Pod manifest

Save as `rccl-pods.yaml`, substituting your node names. Four things are load-bearing:

- **`hostNetwork: true`** — RCCL drives RoCE over the host's `ionic` netdevs; a Calico
  veth cannot see them.
- **`/dev/infiniband` hostPath** — exposes `uverbs0..N` to the container.
- **`sshd` on port 2222** — `hostNetwork` means port 22 already belongs to the host.
- **`/dev/shm` as a sized memory `emptyDir`** — the 64 MB default hangs multi-rank jobs.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: rccl-head
  labels: { app: rccl }
spec:
  restartPolicy: Never
  hostNetwork: true
  hostIPC: true
  dnsPolicy: ClusterFirstWithHostNet
  nodeName: <HEAD-NODE>
  tolerations: [{ operator: Exists }]
  containers:
    - name: rccl
      image: docker.io/rocm/roce-workload:ubuntu24_rocm-7.0.2_rccl-7.0.2_anp-v1.2.0_ainic-1.117.5-a-77
      imagePullPolicy: IfNotPresent
      securityContext:
        privileged: true
        capabilities: { add: ["IPC_LOCK", "SYS_PTRACE", "SYS_NICE"] }
      command: ["bash", "-lc"]
      args:
        - |
          mkdir -p /root/.ssh && cp /ssh-keys/* /root/.ssh/ 2>/dev/null
          cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys
          chmod 700 /root/.ssh && chmod 600 /root/.ssh/id_rsa /root/.ssh/authorized_keys
          printf 'Port 2222\nPermitRootLogin yes\nStrictModes no\n' >> /etc/ssh/sshd_config
          mkdir -p /run/sshd && /usr/sbin/sshd
          echo "READY $(hostname)"
          sleep infinity
      volumeMounts:
        - { name: infiniband, mountPath: /dev/infiniband }
        - { name: dshm,       mountPath: /dev/shm }
        - { name: sshkeys,    mountPath: /ssh-keys, readOnly: true }
      resources:
        limits: { amd.com/gpu: "8" }
  volumes:
    - { name: infiniband, hostPath: { path: /dev/infiniband } }
    - { name: dshm, emptyDir: { medium: Memory, sizeLimit: 32Gi } }
    - { name: sshkeys, secret: { secretName: rccl-ssh, defaultMode: 0600 } }
---
apiVersion: v1
kind: Pod
metadata:
  name: rccl-worker
  labels: { app: rccl }
spec:
  restartPolicy: Never
  hostNetwork: true
  hostIPC: true
  dnsPolicy: ClusterFirstWithHostNet
  nodeName: <WORKER-NODE>
  tolerations: [{ operator: Exists }]
  containers:
    - name: rccl
      image: docker.io/rocm/roce-workload:ubuntu24_rocm-7.0.2_rccl-7.0.2_anp-v1.2.0_ainic-1.117.5-a-77
      imagePullPolicy: IfNotPresent
      securityContext:
        privileged: true
        capabilities: { add: ["IPC_LOCK", "SYS_PTRACE", "SYS_NICE"] }
      command: ["bash", "-lc"]
      args:
        - |
          mkdir -p /root/.ssh && cp /ssh-keys/* /root/.ssh/ 2>/dev/null
          cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys
          chmod 700 /root/.ssh && chmod 600 /root/.ssh/id_rsa /root/.ssh/authorized_keys
          printf 'Port 2222\nPermitRootLogin yes\nStrictModes no\n' >> /etc/ssh/sshd_config
          mkdir -p /run/sshd && /usr/sbin/sshd
          echo "READY $(hostname)"
          sleep infinity
      volumeMounts:
        - { name: infiniband, mountPath: /dev/infiniband }
        - { name: dshm,       mountPath: /dev/shm }
        - { name: sshkeys,    mountPath: /ssh-keys, readOnly: true }
      resources:
        limits: { amd.com/gpu: "8" }
  volumes:
    - { name: infiniband, hostPath: { path: /dev/infiniband } }
    - { name: dshm, emptyDir: { medium: Memory, sizeLimit: 32Gi } }
    - { name: sshkeys, secret: { secretName: rccl-ssh, defaultMode: 0600 } }
```

```bash
kubectl apply -f rccl-pods.yaml
kubectl get pod -l app=rccl -w
```

## Step 5 — Verify RDMA reached the pod

Before touching MPI, confirm the container actually sees the fabric:

```bash
kubectl exec rccl-head -- bash -lc 'ls /sys/class/infiniband; ls /dev/infiniband; ibv_devices'
```

Expect `ionic_0..ionic_8`, `uverbs0..uverbs8`, and a GUID per device. If this is empty,
nothing downstream will work.

## Step 6 — Verify pod-to-pod SSH

The gate for `mpirun`:

```bash
kubectl exec rccl-head -- bash -lc \
  'ssh -o StrictHostKeyChecking=no -p 2222 root@<WORKER-NODE> hostname'
```

## Step 7 — Hostfile and HCA discovery

```bash
kubectl exec rccl-head -- bash -lc '
  printf "<HEAD-NODE> slots=8\n<WORKER-NODE> slots=8\n" > /root/hostfile
  HCA=$(for d in $(ls /sys/class/infiniband | grep ^ionic_ | sort -V); do
          grep -q ACTIVE /sys/class/infiniband/$d/ports/1/state && printf "%s:1," "$d"; done | sed "s/,$//")
  echo "$HCA" > /root/hca_devs
  echo "HCA_DEVS=$HCA"
'
```

## Step 8 — MPI smoke test

Always run this before perf — it isolates launcher problems from RCCL problems:

```bash
kubectl exec rccl-head -- bash -lc '
  export OMPI_MCA_plm_rsh_agent="ssh -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
  /root/ompi/install/bin/mpirun --allow-run-as-root -np 16 -map-by slot --bind-to numa \
    -hostfile /root/hostfile \
    --mca oob_tcp_if_include enp81s0f1 --mca btl_tcp_if_include enp81s0f1 \
    hostname | sort | uniq -c
' < /dev/null
```

Expect 8 ranks per node.

## Step 9 — Run `all_reduce_perf` (native ionic)

Write the launcher to a **file** inside the pod and run it with stdin from `/dev/null`.
Piping a script into `bash` over stdin lets `mpirun` consume the remainder and the job
hangs right after the smoke test.

```bash
kubectl exec rccl-head -- bash -lc 'cat > /root/run.sh <<"EOS"
HCA=$(cat /root/hca_devs)
LIBS=/root/ompi/install/lib:/root/ucx/install/lib:/root/rccl/build/release:/root/amd-anp/build
export OMPI_MCA_plm_rsh_agent="ssh -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

/root/ompi/install/bin/mpirun \
  --allow-run-as-root -np 16 -map-by slot --bind-to numa --hostfile /root/hostfile \
  --mca oob_tcp_if_include enp81s0f1 --mca btl_tcp_if_include enp81s0f1 \
  --mca pml ob1 --mca btl tcp,self,vader --mca coll ^hcoll \
  -x NCCL_IB_HCA=$HCA -x NCCL_SOCKET_IFNAME=enp81s0f1 \
  -x NCCL_MAX_NCHANNELS=56 -x NCCL_ALGO=RING \
  -x NCCL_PXN_DISABLE=0 -x HSA_NO_SCRATCH_RECLAIM=1 \
  -x NCCL_IB_TC=104 -x NCCL_IB_FIFO_TC=192 \
  -x NCCL_NET_OPTIONAL_RECV_COMPLETION=1 -x NCCL_IB_USE_INLINE=1 \
  -x NCCL_GDR_FLUSH_DISABLE=1 -x IONIC_LOCKFREE=all \
  -x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 \
  -x LD_LIBRARY_PATH=$LIBS \
  -x LD_PRELOAD=/root/rccl/build/release/librccl.so.1.0 \
  /root/rccl-tests/build/all_reduce_perf -b 8 -e 16g -f 2 -g 1 -n 20
EOS
bash /root/run.sh < /dev/null' < /dev/null
```

Runtime is roughly 45 seconds. Abridged output:

```
#       size    count  type  redop  root    time   algbw   busbw #wrong
   134217728 33554432 float    sum    -1  1275.7  105.21  197.26      0
   536870912 134217728 float   sum    -1  5029.3  106.75  200.15      0
  1073741824 268435456 float   sum    -1   10045  106.89  200.42      0
 17179869184 4294967296 float  sum    -1  161309  106.50  199.69      0
# Avg bus bandwidth    : 60.3716
```

## Step 10 — Cleanup

```bash
kubectl delete pod rccl-head rccl-worker
kubectl delete secret rccl-ssh
sudo scontrol update NodeName=<head>,<worker> State=RESUME
```

---

## Gotchas

**`--mca btl tcp,self,sm` fails.** The `sm` BTL was removed in Open MPI 3.0 and this
image ships 4.1.6. Use `vader`. Symptom:

```
As of version 3.0.0, the "sm" BTL is no longer available in Open MPI.
mca_bml_base_open() failed --> Returned "Not found" (-13)
```

**`--mca plm_rsh_agent "ssh -p 2222 ..."` fails.** Shell expansion word-splits the value
and `mpirun` parses `-p` as its own flag (`unknown option "-p"`). Export
`OMPI_MCA_plm_rsh_agent` instead.

**`find /` inside the pod hangs.** If NFS is mounted, a full-filesystem search walks it.
Scope searches to `/root /opt /usr`.

**The ANP path segfaults.** Adding `--mca btl ^vader,openib`,
`-x NCCL_NET_PLUGIN=librccl-anp.so`, `-x NCCL_IB_GID_INDEX=1`,
`-x NCCL_DMABUF_ENABLE=1` with `-b 1k -e 16g -f 2 -g 1 -n 10` crashes all ranks on the
*worker* node inside `librccl` under `fgets` — a file open returning NULL that is not
checked. The native path is unaffected. Unresolved.

**Don't co-locate with the control plane.** Running RCCL on the node hosting etcd and
the API server risks disrupting the cluster, and control-plane nodes tend to also host
other long-lived workloads.

## Tuning the 200 GB/s ceiling

Bandwidth pins to almost exactly 200 GB/s from 256 MB through 16 GB — a flat ceiling
rather than a gradual rolloff. Each Pensando NIC is 400 Gb/s ≈ 50 GB/s, so 200 GB/s is
precisely **4 rails at line rate**, suggesting RCCL fans out over only half the NICs.

Three things to try, in order:

1. **Drop the non-GPU-paired NIC.** There are 9 ACTIVE `ionic` devices but only 8 GPUs.
   Passing all nine to `NCCL_IB_HCA` can skew topology detection. Identify the odd one
   with `lstopo-no-graphics --of txt | grep -E "CoProc opencl|OpenFabrics"` — each
   backend NIC sits adjacent to a GPU — and exclude it.
2. **Read the real DSCP values** instead of the `104`/`192` fallback:
   ```bash
   sudo nicctl show qos dscp-to-purpose
   # NCCL_IB_TC      = data_dscp << 2
   # NCCL_IB_FIFO_TC = cts_dscp  << 2
   ```
   Wrong traffic classes mean no priority flow control under congestion.
3. **Count the rails actually in use** with `NCCL_DEBUG=INFO` and grep for `NET/IB`
   lines to confirm how many HCAs RCCL selected.

Also confirm `kernel.numa_balancing=0` on every node — with it enabled, busbw plateaus
well below reference.
