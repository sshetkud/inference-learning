# MaxText Llama 3.1 70B on Kubernetes (4× MI300X, Kubeflow)

Bring-up of Kubeflow 1.10.1 on the lab cluster, then a MaxText Llama 3.1 70B
job pinned to four MI300X nodes. `kubectl` runs on `smc200x-ccs-e12-35`.
Control plane is `smci355-ccs-aus-r16-10` (`https://10.194.134.99:6443`).
Kubeconfig on that host:

```bash
export KUBECONFIG=/etc/kubernetes/kubeconfig-shared
```

Nodes used for the job:

| Node | GPUs | Role |
|---|---|---|
| `dell300x-ccs-aus-k11-03` | 8 | worker |
| `dell300x-ccs-aus-k11-09` | 8 | worker |
| `dell300x-ccs-aus-k11-15` | 8 | worker |
| `dell300x-ccs-aus-k11-21` | 8 | worker |

32 GPUs total, 192 GiB HBM each. Kubernetes v1.31.14.

The training manifest is [`maxtext-llama3.1-70b-k11.yaml`](maxtext-llama3.1-70b-k11.yaml).
It was applied from `smc200x-ccs-e12-35`. Worker index `2` landed in
`Error`; the rest of the Kubeflow platform stayed healthy.

---

## 1. Add the four nodes

On e12-35, print a join command. Run that printed line with `sudo` on each
k11 node. Do not reuse a token from chat; tokens expire.

```bash
export KUBECONFIG=/etc/kubernetes/kubeconfig-shared
kubeadm token create --print-join-command --kubeconfig /etc/kubernetes/kubeconfig-shared
```

On each k11 node, kubeadm preflight needs `conntrack`:

```bash
sudo apt-get install -y conntrack
sudo kubeadm join 10.194.134.99:6443 --token <token> --discovery-token-ca-cert-hash sha256:<hash>
```

Check from e12-35:

```bash
kubectl get nodes -o wide | grep k11
```

All four should be `Ready` with 8 `amd.com/gpu`.

## 2. NFS directory

`storage.cs-aus.dcgpu:/dcgpuval` is mounted on e12-35 at `/mnt/dcgpuval`.
The provisioner writes volumes under this directory:

```bash
mkdir -p /mnt/dcgpuval/afde/sshetkud/kubeflow-pvc
chmod 2775 /mnt/dcgpuval/afde/sshetkud/kubeflow-pvc
```

## 3. NFS provisioner

The upstream example volume (`10.3.243.101:/ifs/kubernetes`) does not mount
from this cluster. Set both the env vars and the pod volume. Setting only
the env leaves the kubelet mounting the example server, and the pod stays
in `ContainerCreating`.

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/nfs-subdir-external-provisioner/master/deploy/rbac.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/nfs-subdir-external-provisioner/master/deploy/deployment.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/nfs-subdir-external-provisioner/master/deploy/class.yaml

kubectl -n default set env deployment/nfs-client-provisioner \
  NFS_SERVER=storage.cs-aus.dcgpu \
  NFS_PATH=/dcgpuval/afde/sshetkud/kubeflow-pvc

kubectl -n default patch deployment nfs-client-provisioner --type=json -p='[
  {"op":"replace","path":"/spec/template/spec/volumes/0/nfs/server","value":"storage.cs-aus.dcgpu"},
  {"op":"replace","path":"/spec/template/spec/volumes/0/nfs/path","value":"/dcgpuval/afde/sshetkud/kubeflow-pvc"}
]'

kubectl -n default rollout status deployment/nfs-client-provisioner
```

`nfs-client-provisioner` should be `1/1 Running`.

## 4. Default StorageClass

`class.yaml` creates `nfs-client`. Mark it default so Kubeflow claims that
omit a class still bind:

```bash
kubectl patch storageclass nfs-client -p \
  '{"metadata":{"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
kubectl get storageclass
```

`nfs-client` is `k8s-sigs.io/nfs-subdir-external-provisioner`, binding
`Immediate`, reclaim `Delete`.

## 5. Kubeflow PVCs

Kubeflow creates these claims. They stay `Pending` until step 3 is done,
then bind on their own. No separate PVC manifest.

```bash
kubectl get pvc -n kubeflow
```

| Claim | Size | Status |
|---|---|---|
| `katib-mysql` | 10Gi | Bound |
| `minio-pvc` | 20Gi | Bound |
| `mysql-pv-claim` | 20Gi | Bound |

`seaweedfs-pvc` (20Gi, Bound) is added in step 8.

## 6. Install Kubeflow 1.10.1

`kubectl kustomize` cannot render the Helm charts in these manifests.
Install standalone Kustomize 5.4.2. Do not install it with snap.

```bash
curl -sL -o /tmp/kustomize.tgz \
  https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2Fv5.4.2/kustomize_v5.4.2_linux_amd64.tar.gz
tar -C /tmp -xzf /tmp/kustomize.tgz
sudo install -m 0755 /tmp/kustomize /usr/local/bin/kustomize

git clone --branch v1.10.1 https://github.com/kubeflow/manifests.git ~/manifests
cd ~/manifests
while ! kustomize build example | kubectl apply --server-side --force-conflicts -f -; do
  echo retrying
  sleep 20
done

kubectl get pods -n kubeflow
kubectl get crd | grep kubeflow
```

This installs Istio, cert-manager, Dex, Pipelines, Katib, KServe, the
Training Operator, and the notebook services. `jaxjobs.kubeflow.org`,
`mpijobs.kubeflow.org`, and `pytorchjobs.kubeflow.org` should exist.
Istio injection stays on the Kubeflow namespaces only.

## 7. Confirm MySQL and Katib

```bash
kubectl get pods -n kubeflow | grep -E 'mysql|katib'
```

After the claims bind, `mysql`, `katib-mysql`, and `katib-db-manager` go
`Running`. MinIO stays down until the next step.

## 8. Replace MinIO with SeaweedFS

The image shipped by Kubeflow 1.10.1,
`gcr.io/ml-pipeline/minio:RELEASE.2019-08-14T20-37-41Z-license-compliance`,
is gone from that registry. `minio/minio` on Docker Hub and `quay.io/minio/minio`
both refuse pulls from this network. The checkout already contains Kubeflow's
SeaweedFS replacement, and `chrislusf/seaweedfs:3.85` does pull.

Use the `istio` overlay. `base` is for a cluster that was installed without
MinIO.

```bash
cd ~/manifests/experimental/seaweedfs
kustomize build istio | kubectl apply --server-side --force-conflicts -f -
kubectl -n kubeflow scale deployment minio --replicas=0
kubectl -n kubeflow rollout status deployment/seaweedfs
kubectl -n kubeflow wait --for=condition=complete job/init-seaweedfs --timeout=300s
```

The overlay redirects `minio-service` at SeaweedFS and also re-applies the
pipeline bundle, which puts the dead MinIO image back. Scaling `minio` to 0
stops that pull. `seaweedfs-pvc` (20Gi) binds on `nfs-client`.

## 9. Restart the pipeline API

`ml-pipeline` crash-loops with `Failed to check if object store bucket exists. Error: 503`
while MinIO is down. Restart it after SeaweedFS is ready.

```bash
kubectl -n kubeflow rollout restart deployment/ml-pipeline
kubectl -n kubeflow rollout status deployment/ml-pipeline
kubectl -n kubeflow exec deploy/ml-pipeline -c ml-pipeline-api-server -- \
  wget -qO- http://localhost:8888/apis/v2beta1/healthz
```

A healthy reply is `{"multi_user":true, ...}`.

## 10. Pin two pods onto the k11 nodes

`metadata-grpc` and `notebook-controller` landed on MI355X nodes where the
kubelet cannot reach pod IPs (`network is unreachable` on the Istio startup
probe). Pin them to the four k11 nodes.

```bash
PATCH='{"spec":{"template":{"spec":{"affinity":{"nodeAffinity":{"requiredDuringSchedulingIgnoredDuringExecution":{"nodeSelectorTerms":[{"matchExpressions":[{"key":"kubernetes.io/hostname","operator":"In","values":["dell300x-ccs-aus-k11-03","dell300x-ccs-aus-k11-09","dell300x-ccs-aus-k11-15","dell300x-ccs-aus-k11-21"]}]}]}}}}}}}'

kubectl -n kubeflow patch deploy metadata-grpc-deployment --type=merge -p "$PATCH"
kubectl -n kubeflow patch deploy notebook-controller-deployment --type=merge -p "$PATCH"
kubectl -n kubeflow rollout status deploy/metadata-grpc-deployment
kubectl -n kubeflow rollout status deploy/notebook-controller-deployment
```

Both should be `2/2 Running`.

## 11. Backup

Taken on e12-35 before the SeaweedFS change and the affinity pin:

```text
~/kf-backup/20261014-191018
```

That directory holds the Kubeflow deployments, services, configmaps, RBAC,
policies, PVC definitions, and secrets from before the change, plus
`RESTORE.sh`.

```bash
bash ~/kf-backup/20261014-191018/RESTORE.sh
```

## 12. MaxText Llama 3.1 70B job

The harness entry point is `_k8s_job.sh`, which reads
`JOB_COMPLETION_INDEX` from a Kubernetes Indexed Job. A Kubeflow `JAXJob`
sets different env vars, so the manifest is a `batch/v1` Job in the
`kubeflow` namespace.

Shape that fits 192 GiB MI300X. The MI355X run (batch 5, sequence 8192) is
sized for 288 GiB and is too large here.

| Setting | Value |
|---|---|
| Model | `llama3.1-70b` |
| Nodes | the four k11 hosts |
| `ici_fsdp_parallelism` | 8 |
| `dcn_fsdp_parallelism` | 4 |
| `per_device_batch_size` | 1 |
| `max_target_length` | 2048 |
| `remat_policy` | `full` |
| dtype | bf16 |
| `dataset_type` | `synthetic` |
| steps | 15 |
| Image | `rocm/jax-training:maxtext-v26.3` |
| Global batch | 32 (65,536 tokens/step) |

32-way FSDP (`ici_fsdp=8` × `dcn_fsdp=4`) puts about 4 GiB of bf16 weights
and about 25 GiB of Adam state on each GPU. Usable pool is about 173 GiB
(`192 × 0.9`).

`rocm/jax-training:maxtext-v26.3` must already be on all four nodes
(about 72 GB). `imagePullPolicy` is `IfNotPresent`. Pods are privileged
and use host networking so RCCL can see the NICs. `/dev/shm` is 64Gi
per pod.

```bash
kubectl apply -f maxtext-llama3.1-70b-k11.yaml
kubectl -n kubeflow get pods -l job-name=maxtext-llama31-70b-k11 -w
kubectl -n kubeflow logs -l job-name=maxtext-llama31-70b-k11 -f --prefix
kubectl -n kubeflow delete job maxtext-llama31-70b-k11
```

Logs are written on the NFS share at
`/mnt/dcgpuval/afde/maxtext-slurm/outputs/`.

---

## Kubeflow platform components (for MaxText on MI355X)

Kubeflow is a collection of components for building and operating
machine-learning workflows on Kubernetes. The platform includes a web
interface, training orchestration, experiment tuning, model serving,
notebooks, and pipelines. You install the components you need; you do not
have to use all of them.

For this AMD MI355X cluster running MaxText, here is what each component
does. `kubectl get pods -n kubeflow` lists these platform pods (plus your
MaxText Indexed Job workers when a job is running).

### How to read pod status

| Column | Meaning |
|---|---|
| `READY 1/1` | One application container, no sidecar. |
| `READY 2/2` or `3/3` | Application container plus Istio / other sidecars. |
| `STATUS Running` | Controllers and services that stay up. |
| `STATUS Completed` | One-shot Job finished successfully (e.g. `init-seaweedfs`). |
| `STATUS Error` | Pod exited non-zero. Investigate; do not assume the platform is down. |

### 1. Kubeflow Dashboard (UI)

**User interface.** The web portal for accessing Kubeflow services such as
notebooks, pipelines, and experiments. It provides a convenient interface,
but it does **not** execute MaxText training itself.

| Your pod | Role |
|---|---|
| `centraldashboard` | Main Kubeflow dashboard |

### 2. Kubeflow Pipelines

**Workflow automation.** Connects multiple steps into an automated workflow,
with dependencies, artifacts, and execution tracking.

Example: prepare data → run MaxText training → evaluate model → save results.

| Your pods | Role |
|---|---|
| `ml-pipeline` | Pipeline API server |
| `ml-pipeline-ui` | Pipelines UI |
| `ml-pipeline-persistenceagent` | Records workflow state |
| `ml-pipeline-scheduledworkflow` | Scheduled / recurring runs |
| `ml-pipeline-viewer-crd` | Artifact viewers |
| `ml-pipeline-visualizationserver` | Pipeline visualizations |
| `cache-server` | Caches completed pipeline steps |
| `workflow-controller` | Argo Workflow controller |
| `kubeflow-pipelines-profile-controller` | Per-profile pipeline access |
| `mysql` / `seaweedfs` / `metadata-*` | Pipeline DB, artifacts, ML Metadata |

### 3. Katib

**Experiment optimization.** Automates hyperparameter tuning by running
experiments with different configurations and comparing results.

For MaxText, it could explore batch size, learning rate, or parallelism
settings, if you define an appropriate objective and search space.

| Your pods | Role |
|---|---|
| `katib-controller` | Experiments / trials |
| `katib-ui` | Katib web UI |
| `katib-db-manager` | Katib DB API |
| `katib-mysql` | Katib database |

### 4. KServe

**Model inference.** Deploys trained models as inference services with
endpoints for applications. It can support scaling and serving workflows
depending on the configured runtime.

After training, you could deploy a model for inference. KServe does **not**
train MaxText models.

| Your pods | Role |
|---|---|
| `kserve-controller-manager` | InferenceServices |
| `kserve-localmodel-controller-manager` | Local model cache |
| `kserve-models-web-app` | Model-serving UI |

### 5. Notebooks

Interactive development environments for writing Python, testing JAX code,
inspecting datasets, and experimenting before launching larger GPU jobs.

| Your pods | Role |
|---|---|
| `jupyter-web-app-deployment` | Notebook UI |
| `notebook-controller-deployment` | Manages Notebook CRs |

### 6. TensorBoard

Visualizes training metrics such as loss, learning rate, throughput, and
other logged values. Useful for comparing runs if your MaxText workflow
exports compatible data.

| Your pods | Role |
|---|---|
| `tensorboard-controller-deployment` | TensorBoard CRs |
| `tensorboards-web-app-deployment` | TensorBoard UI |

### How these components fit into the MaxText workflow

```text
Dashboard / UI          →  User submits or monitors work
        │
Pipelines or training   →  Orchestrates jobs and workflow steps
        │
Kubernetes              →  Schedules worker pods onto GPU nodes
        │
MaxText + JAX + ROCm    →  Executes distributed training on AMD GPUs

Katib                   →  Tune configurations (optional)
TensorBoard             →  View training metrics (optional)
Storage / metadata      →  Keep artifacts and run data
KServe                  →  Serve the trained model (optional)
```

### Which components matter for this work?

| Your goal | Relevant components |
|---|---|
| Run MaxText training | Kubernetes plus a suitable Job or distributed-training controller (`batch/v1` Indexed Job in this lab) |
| Automate training and evaluation | Kubeflow Pipelines |
| Tune MaxText parameters | Katib, if configured for your workload |
| Monitor loss and training metrics | TensorBoard and your metrics stack |
| Develop and test JAX code interactively | Notebooks |
| Serve a trained model | KServe |
| Provide a central web portal | Dashboard |

**Bottom line:** For this MI355X validation workload, Kubeflow’s main value is
coordinating and managing the ML lifecycle. The UI, Katib, KServe, notebooks,
and TensorBoard are separate capabilities — not all prerequisites for running
MaxText. The exact minimum installation depends on which component submits and
manages your training jobs. This lab path uses a plain Kubernetes Indexed Job;
platform pods listed above are **not** required for the training run itself
(see the successful r16 job in §13).

### MaxText worker (not a platform pod)

```text
maxtext-llama31-70b-k11-2-zdkkz   0/1   Error
maxtext-llama31-70b-r16-<index>-xxxxx   Completed   # successful r16 workers
```

Training workers are separate from platform components. Inspect without
deleting the Job:

```bash
kubectl -n kubeflow get pods -l job-name=maxtext-llama31-70b-r16 -o wide
kubectl -n kubeflow logs -l job-name=maxtext-llama31-70b-r16 -f --prefix
```

Harness logs on the NFS share:

```text
/mnt/dcgpuval/afde/maxtext-slurm/outputs/
```

---

## What was verified

- NFS provisioner `1/1 Running`, StorageClass `nfs-client` default.
- `katib-mysql`, `minio-pvc`, `mysql-pv-claim`, `seaweedfs-pvc` Bound.
- SeaweedFS `2/2 Running`, `init-seaweedfs` Complete, MinIO scaled to 0.
- `ml-pipeline` `2/2 Running`; health check returned `multi_user: true`.
- `metadata-grpc` and `notebook-controller` `2/2 Running` on k11 nodes.
- Platform pods in `kubeflow` were `Running` / `Completed` after step 10.
- MaxText Indexed Job was applied; worker `maxtext-llama31-70b-k11-2-zdkkz`
  is `Error` (training workload, not a platform failure).
- **MaxText Llama 3.1 70B on r16 succeeded** on
  `smci355-ccs-aus-r16-{06,18,26,30}` (4× MI355X, ionic). See
  [`maxtext-llama3.1-70b-r16.yaml`](maxtext-llama3.1-70b-r16.yaml).

---

## 13. MaxText Llama 3.1 70B on r16 (4× MI355X) — succeeded

Successful Indexed Job on four **MI355X (gfx950)** nodes with ionic RDMA.
Control plane `r16-10` was **excluded** (slow start caused JAX
`RegisterTask` / `DEADLINE_EXCEEDED`). `r16-22` was skipped (stale Ray
worker holding GPUs).

| Node | GPUs | Role |
|---|---|---|
| `smci355-ccs-aus-r16-06` | 8 | worker |
| `smci355-ccs-aus-r16-18` | 8 | worker |
| `smci355-ccs-aus-r16-26` | 8 | worker |
| `smci355-ccs-aus-r16-30` | 8 | worker |

32 GPUs total, 288 GiB HBM each. Manifest:
[`maxtext-llama3.1-70b-r16.yaml`](maxtext-llama3.1-70b-r16.yaml).
Per-pod entry: [`_k8s_job.sh`](_k8s_job.sh).

| Setting | Value |
|---|---|
| Model | `llama3.1-70b:70b:r16` (short alias keeps log names under 255 B) |
| `per_device_batch_size` | 5 |
| `max_target_length` | 8192 |
| `dcn_fsdp_parallelism` | 4 |
| `ici_fsdp_parallelism` | 8 (from `llama3.1-70b.gpu.yml`) |
| `dataset_type` | synthetic |
| steps | 15 |
| Env | `JAX_USE_SHARDY_PARTITIONER=0`, `RCCL_WARP_SPEED_AUTO=0` |
| Image | `rocm/jax-training:maxtext-v26.3` |
| PVC | `kubeflow/maxtext-harness` → NFS `/dcgpuval/afde/maxtext-slurm` |

```bash
export KUBECONFIG=/etc/kubernetes/kubeconfig-shared
kubectl -n kubeflow delete job maxtext-llama31-70b-r16 --ignore-not-found --wait=true
rm -rf /mnt/dcgpuval/afde/maxtext-slurm/outputs/k8s-r16-70b-r16/.coord
mkdir -p /mnt/dcgpuval/afde/maxtext-slurm/outputs/k8s-r16-70b-r16/.coord
kubectl apply -f maxtext-llama3.1-70b-r16.yaml
kubectl -n kubeflow get pods -l job-name=maxtext-llama31-70b-r16 -o wide -w
kubectl -n kubeflow logs -l job-name=maxtext-llama31-70b-r16 -f --prefix
```

Logs land under `/mnt/dcgpuval/afde/maxtext-slurm/outputs/k8s-r16-*`.

**Lessons from failed attempts:** keep `JOB_NAME` / passthrough short (255 B
path limit); wait for image pull on every node before expecting JAX init;
do not pin training pods on the Kubernetes control-plane node.
