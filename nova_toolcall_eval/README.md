# nova_toolcall_eval — tool-calling evaluation for Keural Nova

Scripts + raw results behind [`../TOOL-CALLING.md`](../TOOL-CALLING.md). All compare
**Keural Nova (v1.0 / v1.1)** against the base **Qwen3.6-35B-A3B**. Run on 2× H200,
`transformers` (hf) backend, deterministic (`do_sample=False`).

| Script | What it measures | Result JSON |
|---|---|---|
| `toolcall_eval.py` | Single-turn tool-call emission, all 3 models (12 queries) | `toolcall_results.json` |
| `confirm_v11.py` | Focused v1.1 vs base, non-thinking, side-by-side | `confirm_v11.json` / `.md` |
| `default_template_test.py` | v1.0 & v1.1 with the **new default template** (no `enable_thinking` passed) | `default_template_results.json` |
| `multiturn_test.py` | Full agentic loop: call → tool result → final answer (RAG / deep-research) | `multiturn_results.json` |

## Headline numbers
- **Single-turn (non-thinking):** base 92% · **Nova v1.1 92%** · Nova v1.0 67%
- **Default template (plain request):** Nova v1.1 91% · v1.0 66% — works out of the box
- **Multi-turn (uses tool result to answer):** **Nova v1.1 6/6**, no looping (base 5/6)

## How to run
```bash
cd "/home/work/keural MoE Pro/nova_toolcall_eval"
PYTHONPATH= PYTHONNOUSERSITE=1 python multiturn_test.py   # etc.
```

Key methodology notes (why early numbers were misleading, now fixed):
1. Qwen emits **XML** tool calls (`<tool_call><function=name>…`), not JSON — the scorer must match XML.
2. In **thinking** mode the model's reasoning can exceed the token budget before the tool call
   appears; tests use **non-thinking** (Nova's native mode) for a fair, representative comparison.
