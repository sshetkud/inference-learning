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

## Kubeflow platform services

`kubectl get pods -n kubeflow` lists platform components, not GPU training
workloads (except your MaxText Indexed Job pods). Most pods are healthy
when `READY` shows `1/1`, `2/2`, or `3/3` and `STATUS` is `Running`.

### How to read the columns

| Column | Meaning |
|---|---|
| `READY 1/1` | One application container, no sidecar. |
| `READY 2/2` or `3/3` | Application container plus Istio / other sidecars. |
| `STATUS Running` | Controllers and services that stay up. |
| `STATUS Completed` | One-shot Job finished successfully (e.g. `init-seaweedfs`). |
| `STATUS Error` | Pod exited non-zero. Investigate; do not assume the platform is down. |
| `RESTARTS` with age ~18h | Historical restarts from the storage bring-up. Not an active crash loop if the pod is `Running` now. |

### User interface

| Pod | Role |
|---|---|
| `centraldashboard` | Main Kubeflow dashboard. |
| `jupyter-web-app-deployment` | Notebook UI. |
| `katib-ui` | Hyperparameter-tuning UI. |
| `kserve-models-web-app` | Model-serving UI. |
| `ml-pipeline-ui` | Pipelines UI. |
| `tensorboards-web-app-deployment` | TensorBoard UI. |
| `volumes-web-app-deployment` | PVC management UI. |

### Kubeflow Pipelines

| Pod | Role |
|---|---|
| `ml-pipeline` | Pipeline API server. |
| `cache-server` | Caches completed pipeline steps. |
| `ml-pipeline-persistenceagent` | Records workflow state in the pipeline DB. |
| `ml-pipeline-scheduledworkflow` | Launches scheduled / recurring runs. |
| `ml-pipeline-viewer-crd` | Manages artifact viewer resources. |
| `ml-pipeline-visualizationserver` | Renders pipeline visualizations. |
| `workflow-controller` | Argo Workflow controller that runs pipeline tasks. |
| `kubeflow-pipelines-profile-controller` | Configures pipeline access for Kubeflow profiles. |

### Pipeline storage and metadata

| Pod | Role |
|---|---|
| `mysql` | Kubeflow Pipelines database. |
| `seaweedfs` | S3-compatible artifact store (replaced the deleted MinIO image). |
| `init-seaweedfs` | One-time SeaweedFS setup Job; `Completed` is expected. |
| `metadata-grpc-deployment` | ML Metadata gRPC server. |
| `metadata-envoy-deployment` | Proxy in front of the metadata service. |
| `metadata-writer` | Writes pipeline execution metadata. |

### Katib

| Pod | Role |
|---|---|
| `katib-controller` | Manages experiments, trials, and suggestions. |
| `katib-db-manager` | Katib database API. |
| `katib-mysql` | Katib database. |
| `katib-ui` | Katib web interface. |

### Notebooks and profiles

| Pod | Role |
|---|---|
| `notebook-controller-deployment` | Manages Kubeflow Notebook CRs. |
| `profiles-deployment` | Creates namespaces and access rules for users. |
| `admission-webhook-deployment` | Validates / mutates resources (including PodDefaults). |
| `pvcviewer-controller-manager` | Launches viewers for PVC contents. |

### Training and serving

| Pod | Role |
|---|---|
| `training-operator` | Manages JAXJob, PyTorchJob, MPIJob, and similar CRs. |
| `kserve-controller-manager` | Manages KServe InferenceServices. |
| `kserve-localmodel-controller-manager` | Manages locally cached serving models. |
| `spark-operator-controller` | Manages Spark applications. |
| `spark-operator-webhook` | Validates / mutates Spark resources. |
| `tensorboard-controller-deployment` | Manages TensorBoard resources. |

### Supporting controller

| Pod | Role |
|---|---|
| `metacontroller-0` | Framework used by Kubeflow controllers to reconcile custom resources. |

### MaxText worker (not a platform pod)

```text
maxtext-llama31-70b-k11-2-zdkkz   0/1   Error
```

This is worker index `2` from the four-node Indexed Job in step 12. It is
not a Kubeflow platform component. Inspect without deleting the Job:

```bash
kubectl -n kubeflow get job maxtext-llama31-70b-k11
kubectl -n kubeflow get pods -l job-name=maxtext-llama31-70b-k11 -o wide
kubectl -n kubeflow logs maxtext-llama31-70b-k11-2-zdkkz
kubectl -n kubeflow describe pod maxtext-llama31-70b-k11-2-zdkkz
```

Harness logs on the NFS share:

```text
/mnt/dcgpuval/afde/maxtext-slurm/outputs/
```

Delete only when you intentionally want to cancel or clean up:

```bash
kubectl -n kubeflow delete job maxtext-llama31-70b-k11
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

## MaxText Llama 3.1 70B on r16 (4x MI355X)

Working Indexed Job manifest (nodes `r16-{06,18,26,30}`, no control-plane):

- [`maxtext-llama3.1-70b-r16.yaml`](maxtext-llama3.1-70b-r16.yaml)

```bash
export KUBECONFIG=/etc/kubernetes/kubeconfig-shared
kubectl -n kubeflow delete job maxtext-llama31-70b-r16 --ignore-not-found --wait=true
kubectl apply -f maxtext-llama3.1-70b-r16.yaml
kubectl -n kubeflow get pods -l job-name=maxtext-llama31-70b-r16 -o wide -w
```
