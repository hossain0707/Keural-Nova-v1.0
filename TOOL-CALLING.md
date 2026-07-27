# Keural Nova — Tool-Calling: issue, root cause, fix, and results

This documents a tool-calling issue found in deployment for **Keural Nova v1.0 / v1.1**, the
root cause, the fix that was applied, and the evaluation that confirmed it. Eval scripts and raw
results are under [`nova_toolcall_eval/`](nova_toolcall_eval/).

## Symptom
When served inside an agent / web-search framework, the model would **loop forever** — repeating
a message like *"the web agent isn't responding, I'll check the URL directly"* — instead of
emitting a tool call. Swapping to the base Qwen model in the **same** environment worked.

## Root cause — thinking-mode default (not a model defect)
Keural Nova was fine-tuned on the **non-thinking** chat path. The chat template's default,
however, was **thinking ON**:

```jinja
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}   {# non-thinking: empty think, answer/tool-call directly #}
{%- else %}
    {{- '<think>\n' }}                 {# thinking = the DEFAULT: opens <think>, model reasons #}
{%- endif %}
```

In **thinking mode**, Nova reasons in prose about which tool to use and then **stops without
emitting the `<tool_call>`** → the agent sees no tool call → infinite loop. Base Qwen tolerates
both modes, which is why "swap the model and it works."

## The fix (config only — no retraining, no weight change)
Flip the template default to **non-thinking**:

- **Before:** `{%- if enable_thinking is defined and enable_thinking is false %}`
- **After:**  `{%- if enable_thinking is not defined or enable_thinking is false %}`

This makes non-thinking the default (empty `<think></think>`), while `enable_thinking=true` still
works if reasoning mode is ever wanted. Applied to **both `chat_template.jinja` and the
`chat_template` field in `tokenizer_config.json`** (kept identical); `tokenizer.json` and the
weights are untouched. Re-uploaded to all Nova repos on `mkd-ai` and `mkd-hossain`.

## Results (base Qwen3.6-35B-A3B vs Keural Nova, non-thinking)

**Single-turn tool-call emission** (12 tool-required queries):

| Model | Emits correct tool call |
|---|---:|
| base Qwen3.6-35B-A3B | 92% (11/12) |
| **Keural Nova v1.1** | **92% (11/12)** — on par with base |
| Keural Nova v1.0 | 67% (8/12) |

**Default template** (plain request, no `enable_thinking` passed — i.e. real deployment):
v1.1 = 91%, v1.0 = 66% → **works out of the box**.

**Multi-turn agentic loop** (call → tool result → final answer; the RAG / deep-research case):
Nova v1.1 correctly **uses the returned tool result to answer, with no looping — 6/6**
(base Qwen 5/6). Example: after `get_weather` returns `{서울, 맑음, 12°C}`, v1.1 replies
*"현재 서울 날씨는 맑고 기온은 12도입니다."*

## Deployment guidance
- **Use Keural Nova v1.1** for tool-calling / agent / RAG / deep-research (matches base Qwen at
  ~91%; v1.0 tool-calls but weaker at ~67%).
- **No serve-command or client changes needed** — the non-thinking default is baked into the
  template. Just **clear the Hugging Face cache on the server and re-pull** the updated model so
  the new template files take effect.
- `enable_thinking=true` remains available per request if reasoning mode is ever desired.

## Caveats (honest)
- Tool-call *emission* is ~91% (v1.1), not 100% — it occasionally answers directly instead of
  calling. On par with base.
- Verified the core single-step loop (call → result → answer). Very long multi-hop chains
  (3+ sequential tools) were not exhaustively tested — validate during rollout.
- v1.0/v1.1 differ: v1.1 is the stronger tool-caller; Korean-quality differences between the two
  are small/mixed. A future version will improve both tool-use and Korean together.
