# MCP Deep Dive — Schema, Sampling, and Resources

Three deeper MCP topics, grounded in the `primus-gpu` server: (a) the exact
`primus_submit_train` `inputSchema`/`outputSchema`, (b) adding **sampling** so the server can
self-diagnose a failed job via the host's LLM, and (c) converting a read **tool** into a
subscribable **resource** for live log streaming.

---

## (a) The exact `primus_submit_train` schema — annotated

The MCP client (Cursor) receives this from `tools/list`. FastMCP generated it from the Python
signature + docstring via `pydantic`. Two parts matter: `arguments` (the JSON Schema for inputs)
and `outputSchema`.

### `arguments` (a.k.a. `inputSchema`) — how each field arises

Every entry is `pydantic` translating one Python parameter:

| Python param | JSON Schema emitted | Rule applied |
|---|---|---|
| `exp_config: str = "…405B…yaml"` | `{"type":"string","default":"…"}` | `str`→`string`; default present ⇒ **not** required |
| `nodes: int = 8` | `{"type":"integer","default":8}` | `int`→`integer` |
| `dcn_data_parallelism: int = -1` | `{"type":"integer","default":-1}` | negative default preserved |
| `dry_run: bool = True` | `{"type":"boolean","default":true}` | `bool`→`boolean` |
| `confirm: bool = False` | `{"type":"boolean","default":false}` | the safety-gate flag |

Key structural facts in the descriptor:

- **`"required"` is absent** from `arguments` — because every parameter has a default. (Contrast:
  an earlier version, before `exp_config` had a default, listed `"required":["exp_config"]`.) So
  the agent can legally call it with zero args.
- **`title`** fields (`"Exp Config"`, etc.) are auto-derived from the param names — pure display
  metadata, not validation.
- The **`description`** string is the docstring verbatim, including the `SAFETY GATE` note. This is
  the only thing the model reads to decide how to call the tool — which is why encoding the
  confirm-gate rule in the docstring actually changes agent behavior.
- Types are shallow: everything is `string`/`integer`/`boolean`. There are **no enums or ranges** —
  e.g. `quantization` accepts any string, `launcher` isn't constrained to `{srun,sbatch}` at the
  schema level; that validation happens inside the function body (`if launcher not in (...)`). You
  could tighten this with `Literal["srun","sbatch"]` in Python → JSON Schema `"enum":[...]`.

### `outputSchema`

```json
"outputSchema": {"type":"object",
  "properties":{"result":{"type":"string"}},
  "required":["result"], "title":"primus_submit_trainOutput"}
```

Because the function is annotated `-> str`, FastMCP wraps it as a structured object with a single
`result` string. Modern MCP returns **both** a human-readable `content` block and this
`structuredContent` for programmatic clients. If it returned a `dict`/`pydantic` model instead of
`str`, this schema would gain typed fields (e.g. `job_id`, `state`) the agent could rely on without
parsing prose.

### The safety gate, in schema terms

There's no MCP "confirmation" primitive — it's implemented purely with `confirm: bool` + logic:
`dry_run=false AND confirm=true` ⇒ submit; otherwise return the `CONFIRMATION REQUIRED` preview. The
schema just exposes the flag; the contract lives in the docstring so the model knows to re-call.

---

## (b) Adding **sampling** so the server can self-diagnose a failed job

**Sampling** = the server calls back into the host's LLM (`sampling/createMessage`) instead of
embedding its own API key. Flow:

```
server tool ── sampling/createMessage (prompt) ──► client (Cursor)
client ── (shows user / applies model prefs) ──► host LLM
client ◀── completion ── server        (server now has an LLM answer)
```

Concretely, add a `primus_diagnose` tool that tails a failed job's log and asks the host model to
root-cause it:

```python
from mcp.server.fastmcp import Context

@_tool
async def primus_diagnose(job_id: str, lines: int = 200, ctx: Context = None) -> str:
    """Tail a failed job's log and ask the host LLM to root-cause it."""
    if not job_id.isdigit():
        return "ERROR: job_id must be numeric"
    log = tail_log_for_job(job_id, lines)          # existing backend helper

    result = await ctx.session.create_message(      # <-- the sampling call
        messages=[{
            "role": "user",
            "content": {"type": "text",
                "text": f"This Primus/MaxText job failed. Give the top root cause "
                        f"and one fix.\n\n=== log tail ===\n{log}"}
        }],
        max_tokens=400,
        system_prompt="You are an expert in JAX/XLA, RCCL/ionic RDMA, and Slurm on MI355X.",
        model_preferences={"hints": [{"name": "claude"}],   # host picks the actual model
                           "intelligencePriority": 0.8},
    )
    return result.content.text
```

Why this is powerful and the caveats:

- **No key management** — the server borrows the host's model; the user stays in control (the spec
  has the host mediate/approve sampling, and can show the prompt).
- **Capability-gated** — only works if the client advertised `sampling` at `initialize`. Handle the
  "not supported" case (fall back to returning the raw log).
- **`model_preferences`** are hints, not guarantees — the host chooses. Don't hard-depend on a model.
- Requires the **async** FastMCP handler + a `Context` param (FastMCP injects it). Heavier than the
  current synchronous, text-only tools.
- Trust boundary: you're feeding log contents into an LLM round-trip — fine here, but be mindful for
  anything sensitive.

This turns "job vanished from the queue" into an automatic "likely OOM at step 0 because … — try
`mem_fraction=0.90`" without reading logs by hand.

---

## (c) Converting a read tool into a **resource** with subscriptions (live log streaming)

Today `primus_tail_log(job_id, lines)` is a **tool** — model-controlled, pull-only. A **resource**
is application-controlled, URI-addressed, read-only, and can **push** updates via subscriptions.
Ideal for a live log.

### Static/templated resource

```python
@mcp.resource("log://primus/{job_id}")
def primus_log_resource(job_id: str) -> str:
    """Live tail of a Primus job's sbatch log (read-only)."""
    return tail_log_for_job(job_id, 200)
```

- The `{job_id}` makes it a **resource template**: the client can enumerate a pattern and read
  `log://primus/7407` on demand.
- Appears in `resources/list` (or via templates), fetched with `resources/read`. Semantically
  GET-like: **no side effects**, so the host can auto-attach it to context without treating it as an
  agent action.

### Adding subscriptions (the "live streaming" part)

1. Advertise capability at startup: `resources: { "subscribe": true, "listChanged": true }`.
2. Client sends `resources/subscribe {uri:"log://primus/7407"}`.
3. The server watches the file (an async task tailing NFS) and, on new bytes, emits:
   ```json
   {"jsonrpc":"2.0","method":"notifications/resources/updated",
    "params":{"uri":"log://primus/7407"}}
   ```
4. Client re-reads the resource → user sees the log advance **without polling**.

```python
# sketch of the notifier side
async def _watch_log(job_id, session):
    last = 0
    while job_still_running(job_id):
        size = os.path.getsize(logpath(job_id))
        if size != last:
            last = size
            await session.send_resource_updated(f"log://primus/{job_id}")
        await asyncio.sleep(2)
```

### Tool vs. resource — when to use which

| | Tool (`primus_tail_log`) | Resource (`log://primus/{id}`) |
|---|---|---|
| Control | Model decides to call | App/user attaches; can subscribe |
| Semantics | Action (may have effects) | Read-only data |
| Updates | Pull each call | **Push** via `notifications/resources/updated` |
| Best for | "tail 60 lines now" | "keep this log live in the panel" |

Trade-offs: resources need an async server and a background watcher; client support for
auto-refreshing subscribed resources varies. For a one-shot glance, the tool is simpler; for a
persistent live view, the resource+subscription is the right primitive.

---

## Net summary

- **(a)** The schema is auto-derived, all-optional (no `required`), string/int/bool only, with the
  confirm-gate expressed as a `bool` + docstring contract and a single-string `outputSchema`.
- **(b)** Sampling lets the server "borrow" the host's LLM to self-diagnose failures — key-free but
  async and capability-gated.
- **(c)** Promoting the log tail to a subscribable resource gives push-based live streaming instead
  of pull.

## Related docs

- [primus-gpu-mcp-server-design.md](primus-gpu-mcp-server-design.md) — server design & architecture.
- [primus-gpu-server.py](primus-gpu-server.py) — committed source snapshot.
