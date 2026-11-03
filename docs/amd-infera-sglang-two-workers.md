# AMD InferA — two SGLang workers (Qwen3-0.6B)

Bring-up on `smci355-ccs-aus-r16-10` (2026-11-03). Two InferA SGLang workers, one per GPU, registering into a private etcd. The engines start. Discovery only works against a plaintext etcd that is **not** the Kubernetes etcd on port 2379.

Node-local time in the logs is 2026-11-03 04:23. Commands were run from the InferA environment (`/opt/infera`); that shell shares the host network and does not see host PIDs.

---

## 1. What we wanted

| Item | Value |
|---|---|
| Model | `Qwen/Qwen3-0.6B` |
| Engine | `python -m infera.engine.sglang` |
| GPUs | `HIP_VISIBLE_DEVICES=0` and `1` |
| API ports | `30000` and `30001` |
| Discovery | etcd, prefix `/infera/workers/` |
| Request path | HTTP |
| KV events | ZMQ |

---

## 2. First launch failed at etcd registration

Both workers were started against `127.0.0.1:2379`:

```bash
HIP_VISIBLE_DEVICES=0 python -m infera.engine.sglang \
  --model-path Qwen/Qwen3-0.6B --port 30000 --host 0.0.0.0 \
  --etcd-endpoint 127.0.0.1:2379 \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq

HIP_VISIBLE_DEVICES=1 python -m infera.engine.sglang \
  --model-path Qwen/Qwen3-0.6B --port 30001 --host 0.0.0.0 \
  --etcd-endpoint 127.0.0.1:2379 \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq
```

SGLang itself came up. On the port-30001 process the log showed:

- `The server is fired up and ready to roll!`
- `GET /health` → `200 OK`
- `worker ready: model=Qwen/Qwen3-0.6B url=http://0.0.0.0:30001 engine=EngineType.SGLANG disagg=DisaggMode.MIXED`
- `KvEventPublisher started: endpoint=tcp://0.0.0.0:5557`
- `WARNING: no RadixCache instance available; probe will not emit device-tier events. Snapshot reconciliation still works once the engine emits events on its own.`
- `KV plane up: events_bind=tcp://0.0.0.0:5557 snapshot=http://0.0.0.0:8801 engine_block_size=1 index_block_size=64`
- `using etcd registration: endpoint=127.0.0.1:2379 prefix=/infera/workers/`

Registration then failed and InferA tore the engine down:

```text
ERROR:__main__:worker failed after engine start; tearing down
httpx.ReadError
  File ".../infera/common/registration.py", line 76, in register
    r = await self._http.post("/v3/lease/grant", json={"TTL": self._lease_ttl})
```

The scheduler received `SIGTERM` (exit `-15`) and the process was killed. The GPU/SGLang startup was healthy. The process died because the etcd lease grant never returned an HTTP body.

---

## 3. Port 2379 is the Kubernetes etcd

`etcdctl` is not installed in that shell. Plain HTTP connects and then gets an empty reply:

```text
curl -v http://127.0.0.1:2379/health
curl: (52) Empty reply from server
```

HTTPS completes a TLS 1.3 handshake and then the server closes the connection (`curl: (16) OpenSSL SSL_write: Broken pipe`). The handshake shows:

- server certificate `CN=smci355-ccs-aus-r16-10`
- issuer `CN=etcd-ca`
- the server sends `Request CERT` (client certificate required)

Listeners:

```text
10.194.134.99:2379
127.0.0.1:2379
```

That is the kubeadm etcd client port (mutual TLS). InferA's registration client speaks plain HTTP (`httpcore` HTTP/1.1), so every lease grant fails the same way. Leave this etcd alone. Do not register InferA workers into the cluster etcd.

`ss` printed no PID for those sockets because the InferA shell is a container PID namespace on the host network.

---

## 4. Working setup: private etcd on 2479

A separate plaintext etcd for InferA listens on `127.0.0.1:2479`. One way to start it on the host, without touching port 2379:

```bash
docker run -d --name infera-etcd --network host \
  quay.io/coreos/etcd:v3.5.17 \
  etcd \
  --name infera \
  --data-dir /etcd-data \
  --listen-client-urls http://127.0.0.1:2479 \
  --advertise-client-urls http://127.0.0.1:2479 \
  --listen-peer-urls http://127.0.0.1:2480 \
  --initial-advertise-peer-urls http://127.0.0.1:2480 \
  --initial-cluster infera=http://127.0.0.1:2480
```

Check before starting workers:

```bash
curl -sS http://127.0.0.1:2479/health
curl -sS -X POST http://127.0.0.1:2479/v3/lease/grant \
  -H 'Content-Type: application/json' -d '{"TTL":60}'
```

`/health` should be JSON (`{"health":"true"}`). The lease grant should return a lease id.

Relaunch with that endpoint:

```bash
HIP_VISIBLE_DEVICES=0 python -m infera.engine.sglang \
  --model-path Qwen/Qwen3-0.6B --port 30000 --host 0.0.0.0 \
  --etcd-endpoint 127.0.0.1:2479 \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq

HIP_VISIBLE_DEVICES=1 python -m infera.engine.sglang \
  --model-path Qwen/Qwen3-0.6B --port 30001 --host 0.0.0.0 \
  --etcd-endpoint 127.0.0.1:2479 \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq
```

---

## 5. Result: both workers registered

`POST /v3/kv/range` on prefix `/infera/workers/` returned `"count": "2"` (cluster revision 3).

```bash
curl -sS -X POST http://127.0.0.1:2479/v3/kv/range \
  -H 'Content-Type: application/json' \
  -d '{"key":"L2luZmVyYS93b3JrZXJzLw==","range_end":"L2luZmVyYS93b3JrZXJzMA=="}'
```

The two keys are `/infera/workers/0.0.0.0:30000` and `/infera/workers/0.0.0.0:30001`. Decoded values:

| Field | Worker 0 | Worker 1 |
|---|---|---|
| `worker_id` | `0.0.0.0:30000` | `0.0.0.0:30001` |
| `url` | `http://0.0.0.0:30000` | `http://0.0.0.0:30001` |
| `model_name` | `Qwen/Qwen3-0.6B` | `Qwen/Qwen3-0.6B` |
| `engine` | `sglang` | `sglang` |
| `disagg_mode` | `mixed` | `mixed` |
| `request_transport` | `http` | `http` |
| `kv_events_endpoint` | `tcp://0.0.0.0:58345` | `tcp://0.0.0.0:43671` |
| `kv` record | present | **absent** |
| lease | set | set |

Worker 0's `kv` record:

| Field | Value |
|---|---|
| `engine_block_size` | 1 |
| `index_block_size` | 64 |
| `tokenizer` | `Qwen/Qwen3-0.6B` |
| `tokenizer_digest` | `aeb13307a71acd8f` |
| `supports_events` | true |
| `event_version` | 1 |
| `events_endpoint` | `tcp://0.0.0.0:5557` |
| `snapshot_endpoint` | `http://0.0.0.0:8801` |
| `tiers` | `["device"]` |

Worker 1 registered with no `kv` object. Both processes use the default KV ports `5557` (events) and `8801` (snapshot), so the second worker cannot bind them. Give each worker its own KV event and snapshot ports (`python -m infera.engine.sglang --help`).

Both records advertise `0.0.0.0`. That address is valid as a bind address and is not a routable address for another host. Pass the node's real IP (`10.194.134.99` on this box) as `--host`.

---

## 6. How to check the servers

There is no router on port 8000 in this run. Each worker serves HTTP on its own port.

```bash
curl -sf http://127.0.0.1:30000/health && echo worker0 OK
curl -sf http://127.0.0.1:30001/health && echo worker1 OK
```

A chat completion goes to a worker port, not `localhost:8000`. This call has not been captured yet in this session:

```bash
curl -sS http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B",
       "messages":[{"role":"user","content":"1+1=?"}],
       "max_tokens":50}'
```

Use `30001` for the second worker. `localhost:8000/v1/chat/completions` and `localhost:8000/v1/workers` apply only after a separate InferA router is started in front of these workers.

---

## 7. Open items

1. Restart both workers with `--host 10.194.134.99` (or the current node IP) so the registered URL is reachable from other hosts.
2. Give worker 1 distinct KV event and snapshot ports, then confirm its etcd value contains a `kv` object like worker 0.
3. Run the `/health` and `/v1/chat/completions` checks above and keep the response.
4. The first (failed) start logged `no RadixCache instance available`. Re-check that line after the KV-port fix. Worker 0's registered record already says `supports_events: true` with a device tier.
