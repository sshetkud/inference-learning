# MCP Deep Dive — Fundamentals

How the Model Context Protocol actually works under the hood, grounded in the `primus-gpu` and
`lab-gpu` servers: architecture & roles, the JSON-RPC wire protocol, the connection lifecycle, the
three server primitives, client→server capabilities, transports, how FastMCP implements it, the
security model, and an end-to-end trace of a single call.

For the schema/sampling/resources follow-ups, see
[mcp-deep-dive-schema-sampling-resources.md](mcp-deep-dive-schema-sampling-resources.md).

---

## 1. Architecture & roles

MCP has three distinct roles:

- **Host** — the application the user interacts with (Cursor). It embeds one or more clients and
  enforces policy/consent.
- **Client** — a connector inside the host; maintains a **1:1 stateful session** with exactly one
  server.
- **Server** — a process exposing capabilities (`server.py`). Can be local (subprocess) or remote.

Key property: a client↔server session is **stateful and long-lived**, not request/response like
REST. Capabilities are negotiated once, then the session stays open.

---

## 2. Wire protocol: JSON-RPC 2.0

Every message is one of three shapes:

**Request** (expects a response, has an `id`):

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call",
 "params":{"name":"slurm_job_status","arguments":{"job_id":"7407"}}}
```

**Response** (`result` or `error`, same `id`):

```json
{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"7407 RUNNING …"}]}}
```

**Notification** (no `id`, fire-and-forget):

```json
{"jsonrpc":"2.0","method":"notifications/tools/list_changed"}
```

Errors use standard JSON-RPC codes (`-32601` method not found, `-32602` invalid params, etc.). Note
there are **two error channels**: protocol errors (JSON-RPC `error`) vs. **tool execution errors**,
which come back as a normal `result` with `isError: true` — that's by design, so the model can see
and reason about the failure. The `_tool` decorator implements exactly this: it catches exceptions
and returns them as text rather than crashing the session.

---

## 3. Connection lifecycle

```
1. initialize (request)     client → server   protocolVersion + client capabilities
   initialize (response)    server → client   protocolVersion + server capabilities + serverInfo
2. notifications/initialized  client → server  "ready"
3. … normal operation: tools/list, tools/call, resources/read, …
4. shutdown (transport close)
```

During capability negotiation each side declares what it supports, e.g. a server advertising tools
with live updates:

```json
"capabilities": { "tools": { "listChanged": true },
                  "resources": { "subscribe": true, "listChanged": true } }
```

The agent only attempts what was negotiated — this is how it knows, at runtime, that `primus-gpu`
has tools but no prompts/resources.

---

## 4. The three server primitives in depth

### Tools (model-controlled)

`tools/list` returns each tool's `name`, `description`, and `inputSchema` (JSON Schema). FastMCP
builds the schema from the Python signature:

```python
@_tool
def primus_submit_train(exp_config: str, nodes: int = 8,
                        steps: int = 50, dry_run: bool = True) -> str: ...
```

→ becomes

```json
{"name":"primus_submit_train",
 "inputSchema":{"type":"object",
   "properties":{"exp_config":{"type":"string"},
                 "nodes":{"type":"integer","default":8},
                 "steps":{"type":"integer","default":50},
                 "dry_run":{"type":"boolean","default":true}},
   "required":["exp_config"]}}
```

The docstring becomes the `description` the model reads. Tools can also declare an `outputSchema`
for structured results (the descriptors show `outputSchema` with a `result` string).

### Resources (application/user-controlled, read-only)

Identified by URI (`log://vllm/{job_id}`). Two access patterns: `resources/list` (direct) and
**resource templates** (parameterized URIs). Clients can `resources/subscribe` to get
`notifications/resources/updated` when data changes. Unlike tools, resources are meant to be **read**
(no side effects) — think GET vs POST.

### Prompts (user-controlled)

Reusable, parameterized templates surfaced as UI affordances (e.g. slash-commands). `prompts/list` /
`prompts/get`. Neither of these servers uses prompts.

---

## 5. Client→server capabilities (the reverse direction)

MCP isn't just server→client. A server can call back into the client for:

- **Sampling** (`sampling/createMessage`) — the server asks the host's LLM to generate a completion.
  This lets a server be "agentic" without holding its own API key. The host mediates (shows the
  user, applies model prefs). Powerful but neither server uses it (yet).
- **Roots** (`roots/list`) — the client tells the server which filesystem/URI roots it's allowed to
  operate within (a sandboxing boundary).
- **Elicitation** (newer) — the server requests structured input from the user mid-operation.

These servers instead implement their own confirmation pattern **in-band** (the `confirm=true` gate
on `primus_submit_train`), a pragmatic alternative to elicitation.

---

## 6. Transports

| Transport | Use | Ours |
|---|---|---|
| **stdio** | Local subprocess; JSON-RPC over stdin/stdout; newline-delimited | ✅ — Cursor spawns `ssh … python server.py`, pipes stdio through the SSH tunnel |
| **Streamable HTTP** (current) | Remote servers; single `/mcp` endpoint, optional SSE upgrade for streaming/server-initiated msgs | — |
| HTTP+SSE (legacy) | Older two-endpoint remote transport, now superseded | — |

The setup is a neat trick: a **local stdio server that happens to run on a remote host** — the SSH
tunnel makes the jump-host process behave like a local subprocess to Cursor. That's why
`backend.py` can use plain subprocesses (no nested auth) once it's on the jump host.

---

## 7. How FastMCP (the `mcp` Python SDK) implements all this

```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("primus-gpu", instructions="…")   # server metadata + capabilities

@mcp.tool()                 # registers into an in-memory tool registry
def some_tool(x: int) -> str: ...

mcp.run()                   # sets up stdio transport, serves the JSON-RPC loop
```

Under the hood FastMCP: introspects signatures/type hints (via `pydantic`) → JSON Schema; wires
`tools/list` and `tools/call` to the registry; serializes returns into content blocks. Because the
registry is built **at import time** and `mcp.run()` then blocks, editing `server.py` on disk changes
nothing until the process is respawned — the **reload requirement**.

---

## 8. Security model (why "read-mostly / verify-first" matters)

MCP's spec puts user consent and control front and center. Practical implications these servers
follow:

- The server runs as **your account**, inheriting your Slurm/SSH permissions and Conductor
  reservation — so it can do anything you can. Hence the discipline: metrics/log tools only
  `curl`/`grep`/`tail`; write tools verify state first; destructive actions are gated.
- Inputs are validated (`job_id.isdigit()`) and shell-interpolated values are `shlex.quote()`-escaped
  to prevent command injection — critical because tool args originate from an LLM.
- The host (Cursor) is the **consent boundary**: it can prompt before a `tools/call`, and it decides
  which servers are enabled.

---

## 9. End-to-end trace of one call (`slurm_job_status`)

```
Cursor: user asks "status of 7407"
  → model emits tools/call {name:"slurm_job_status", arguments:{job_id:"7407"}}
  → JSON-RPC frame written to stdin of the ssh-tunneled python process
server.py: _tool wrapper → slurm_job_status("7407")
  → jid.isdigit() ✓
  → backend.ssh_run("squeue -j 7407 …")  (subprocess on jump host)
  → formats squeue+sacct text
  → returns {"content":[{"type":"text","text":"7407 … RUNNING …"}]}
Cursor: injects that text into the model's context → model answers the user
```

---

## Where to go next

- **(a)** the exact JSON of `primus_submit_train`'s `inputSchema`/`outputSchema`,
- **(b)** adding **sampling** so the server can self-diagnose a failed job via the host's LLM,
- **(c)** converting a read tool into a proper **resource with subscriptions** for live log streaming.

All three are covered in
[mcp-deep-dive-schema-sampling-resources.md](mcp-deep-dive-schema-sampling-resources.md).

## Related docs

- [primus-gpu-mcp-server-design.md](primus-gpu-mcp-server-design.md) — server design & architecture.
- [primus-gpu-server.py](primus-gpu-server.py) — committed source snapshot.
