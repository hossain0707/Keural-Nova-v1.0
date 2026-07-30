# Keural Nova — tool-calling SFT slice

Two JSONL files that teach Qwen3.6-35B-A3B (project "Keural Nova") to emit Qwen
tool-call XML, use tool results, and *not* over-call when no tool is needed.

| file | rows | what |
|---|---:|---|
| `tooluse_short.jsonl` | 14,337 | mixed single-turn / multi-turn / negative tool-use tasks |
| `tooluse_long32k.jsonl` | 3,000 | same tasks embedded in 8k–32k-token context (needle-in-haystack) |

Regenerate both with:

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 HF_XET_HIGH_PERFORMANCE=1 \
  /home/work/.keural_venv/bin/python "scripts/build_tooluse_slice.py"
# quick end-to-end check: add --smoke
```

`data/` is git-ignored (regenerable); the builder script is the tracked artifact.

## Row schema (IMPORTANT — this is ms-swift 4.4.0 NATIVE, not OpenAI)

One JSON object per line:

```json
{
  "messages": [
    {"role": "user",      "content": "지금 서울 날씨 어때?"},
    {"role": "tool_call", "content": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Seoul\"}}"},
    {"role": "tool",      "content": "{\"city\": \"Seoul\", \"condition\": \"clear\", \"temp_c\": 12}"},
    {"role": "assistant", "content": "현재 서울 날씨는 맑고 기온은 12도입니다."}
  ],
  "tools": "[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", ...}}]"
}
```

Rules, all verified against `outputs/merged/Keural-35B-A3B` + ms-swift 4.4.0:

- **Roles**: `system`, `user`, `assistant`, `tool_call`, `tool`. No empty content.
- **A tool call is its own message** with `role: "tool_call"` and `content` = a JSON
  **string** `{"name": ..., "arguments": {…}}` (`arguments` is an object).
  ms-swift does **not** read the OpenAI `tool_calls` field — its dataset
  preprocessor (`swift/dataset/preprocessor/core.py::_check_messages`) pops every
  message key except `{role, content, loss, loss_scale}`, so a `tool_calls` field
  is silently dropped. Tool calls MUST be `tool_call`-role messages.
- **Tool results** use `role: "tool"` (equivalent to `tool_response`); content is a string.
- **`tools` is a per-row top-level field, stored as a JSON string** (a list of
  OpenAI wrapped schemas `{"type":"function","function":{name,description,parameters}}`;
  `"[]"` when the row has no tools). It is a **string** on purpose: nested tool
  objects have per-row-varying `parameters` shapes that break `datasets`/pyarrow
  schema inference. ms-swift `Template._preprocess_tools` parses the string back to
  a list.
- **No `<think>` scaffold is written into content.** For non-thinking (the mode
  Nova is trained/served in), the `qwen3_5` template injects the empty
  `<think>\n\n</think>` itself and `train_full.yaml`'s
  `loss_scale: ignore_empty_think` masks it. (Putting `<think>` into a `tool_call`
  JSON string would also break parsing.)

## How the XML gets rendered (why this schema is correct)

`Qwen/Qwen3.6-35B-A3B` → ms-swift `TemplateType.qwen3_5` → `agent_template = 'qwen3_5'`
(auto-selected; **no `--agent_template` flag needed**). That agent template renders

```
<tool_call>
<function=NAME>
<parameter=P>
VAL
</parameter>
</function>
</tool_call>
```

and `<tool_response>…</tool_response>` for tool results — **byte-identical** to the
model's own `chat_template.jinja`. Both paths were rendered and diffed; a full
ms-swift `train`-mode `encode()` of stored rows confirms the tool-call XML is a
trained (unmasked) target while the empty-`<think>` scaffold and user/tool turns
are masked.

## Trainer config (for whoever wires these into a run)

- `loss_scale: ignore_empty_think` (already set in `configs/train_full.yaml`).
- No agent-template override; `qwen3_5` is auto-selected for this model.
- `max_length` must be raised to `32768` for the long file (short file fits in 8192).
- The `tools` field flows through ms-swift's `messages` preprocessor unchanged.

## Sources & licenses (all commercially usable)

| source | rows used | license | notes |
|---|---:|---|---|
| `glaiveai/glaive-function-calling-v2` @ `e7f4b64…` | 11,100 | Apache-2.0 | single/multi-turn + natural negatives |
| `NousResearch/hermes-function-calling-v1` @ `dae3e1d…` | 2,037 | Apache-2.0 | multi-tool / agentic variety |
| hand-built Korean (weather/stock/search/calculator) | 1,200 | project-owned | no open KO function-calling data exists |
| long-context (synthetic needle-in-log) | 3,000 | project-owned | filler is generated, no external text |

**Excluded:** `Salesforce/xlam-function-calling-60k` — gated on the Hub (kept the
pipeline non-gated / reproducible).

## Mix (short file)

| bucket | rows |
|---|---:|
| glaive single-turn (user → tool_call) | 3,900 |
| glaive multi-turn (call → result → final answer) | 3,900 |
| glaive negative, tools present, answered directly | 2,400 |
| glaive negative, no tools | 900 |
| hermes (single + multi-turn, multi-tool) | 2,037 |
| Korean (single / multi / negative) | 1,200 |

Long file: 3,000 rows, token lengths min 8,605 / median 20,482 / max 31,983 (cap 32,768),
each requiring the model to find an identifier buried in a long operations log and
call the right tool with it.
