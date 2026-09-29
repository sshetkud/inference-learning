# Kubernetes node reservation (taint + label + CRD/RBAC)

How a GPU node is "reserved" for one user on the lab Kubernetes cluster so that
only that user's pods land on it. Used to hold `dell-ccs-e14-34` for the
[vLLM Qwen serve + benchmark](k8s-vllm-qwen-serve-bench.md).

Manifests: [`manifests/reservation-crd.yaml`](manifests/reservation-crd.yaml),
[`manifests/reservation-rbac.yaml`](manifests/reservation-rbac.yaml),
[`manifests/reservation-cr.yaml`](manifests/reservation-cr.yaml).

---

## The mechanism = 3 parts

A reservation is **enforced by the node itself**, not by the CRD. Three pieces work
together:

| Part | Object | Role |
|---|---|---|
| **Enforce** | node **taint** `reservation.lab.amd/owner=sshetkud:NoSchedule` | repels every pod that does *not* tolerate it |
| **Target** | node **label** `reservation.lab.amd/owner=sshetkud` | lets the owner's pods select the node |
| **Record / ACL** | **`Reservation` CR** (CRD `reservations.lab.amd`) | book-of-record: who owns what, when, why; RBAC controls who can create it |

A pod claims a reserved node by opting in to **both** the label (via
`nodeSelector`) and the taint (via a matching `toleration`):

```yaml
nodeSelector:
  reservation.lab.amd/owner: sshetkud
tolerations:
- key: reservation.lab.amd/owner
  operator: Equal
  value: sshetkud
  effect: NoSchedule
```

Any other user's pods lack the toleration, so the `NoSchedule` taint keeps them
off the node — that is what makes the node "reserved".

---

## Enforce a reservation (the commands that actually matter)

```bash
# 1) taint the node so foreign pods are repelled
kubectl taint nodes dell-ccs-e14-34 reservation.lab.amd/owner=sshetkud:NoSchedule

# 2) label the node so the owner's pods can target it
kubectl label nodes dell-ccs-e14-34 reservation.lab.amd/owner=sshetkud

# verify
kubectl describe node dell-ccs-e14-34 | grep -i -A2 taint
kubectl get node dell-ccs-e14-34 --show-labels
```

Release it:

```bash
kubectl taint nodes dell-ccs-e14-34 reservation.lab.amd/owner=sshetkud:NoSchedule-
kubectl label nodes dell-ccs-e14-34 reservation.lab.amd/owner-
```

> The `Reservation` CR is a record/ACL only — creating or deleting it does **not**
> taint/label the node by itself (unless you also run a controller). Always pair
> the CR with the taint+label commands above.

---

## CRD — the book of record

Install [`manifests/reservation-crd.yaml`](manifests/reservation-crd.yaml):

```bash
kubectl apply -f manifests/reservation-crd.yaml
```

- Group `lab.amd`, **cluster-scoped**, kind `Reservation`, shortName `rsv`.
- `spec.owner` (required), `spec.nodes` (required array), `spec.start`/`spec.end`
  (`date-time`), `spec.reason`; `status.phase`
  (`Pending|Active|Expired|Released`) as a status subresource.
- Printer columns: Owner / Nodes / Start / End / Phase, so `kubectl get rsv` is
  readable at a glance.

Create a reservation record ([`manifests/reservation-cr.yaml`](manifests/reservation-cr.yaml)):

```bash
kubectl apply -f manifests/reservation-cr.yaml
kubectl get rsv
```

---

## RBAC — who can reserve

Install [`manifests/reservation-rbac.yaml`](manifests/reservation-rbac.yaml):

```bash
kubectl apply -f manifests/reservation-rbac.yaml
```

| ClusterRole | Verbs | Bound to |
|---|---|---|
| `reservation-admin` | full CRUD on `reservations` + `reservations/status`, **plus** `nodes` get/list/watch/patch/update (to place the taint/label) | User `sshetkud` |
| `reservation-viewer` | read-only `reservations` | Group `system:authenticated` (everyone) |

The `nodes … patch/update` verbs in `reservation-admin` are what allow the owner
to actually taint/label the node — the CR permissions alone can't enforce a
reservation.

---

## TL;DR

`taint` (enforce) + `label` (target) + `Reservation` CR (record, gated by RBAC).
Pods opt in with a matching `nodeSelector` + `toleration`; everyone else is
repelled by the `NoSchedule` taint.
